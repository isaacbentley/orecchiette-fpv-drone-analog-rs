use ort::{inputs, session::Session, value::Tensor};

pub struct NeuralRestorer {
    session: Session,
    /// Whether the loaded model takes the CNR-conditioning `noise`
    /// input. Older 2-input models (image + hidden) predate the
    /// noise-plane channel; feeding them a `noise` input would error, so
    /// it's supplied only when the model actually declares it. Detected
    /// once at load.
    has_noise_input: bool,
    /// Channel count of the model's `hidden_in`/`hidden_out` state, read
    /// from the model itself rather than hardcoded — different training
    /// runs use different widths (the shipped model is 16), and a
    /// mismatch is a hard ONNX dimension error.
    hidden_channels: usize,
}

impl NeuralRestorer {
    pub fn new(model_path: &str, use_gpu: bool) -> ort::Result<Self> {
        // `mut` is only exercised on macOS (the CoreML EP reassigns
        // `builder`); on other targets that block is cfg'd out.
        #[cfg_attr(not(target_os = "macos"), allow(unused_mut))]
        let mut builder = Session::builder()?;

        // CoreML dispatches to the Apple Neural Engine / GPU; it's
        // Apple-only, so on every other target `use_gpu` simply falls
        // through to `ort`'s default CPU provider (nothing to configure).
        #[cfg(target_os = "macos")]
        if use_gpu {
            let eps: Vec<ort::execution_providers::ExecutionProviderDispatch> =
                vec![ort::ep::CoreML::default().build()];
            builder = builder.with_execution_providers(eps)?;
        }
        #[cfg(not(target_os = "macos"))]
        let _ = use_gpu;

        let num_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        let session = builder
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .with_intra_threads(num_cpus)?
            .with_inter_threads(num_cpus)?
            .commit_from_file(model_path)?;
        let has_noise_input = session.inputs().iter().any(|i| i.name() == "noise");
        // `hidden_in` shape is [1, C, H, W] with C fixed; read C from the
        // model. Fall back to 16 (the current model / training default).
        let hidden_channels = session
            .inputs()
            .iter()
            .find(|i| i.name() == "hidden_in")
            .and_then(|i| i.dtype().tensor_shape())
            .and_then(|s| s.get(1).copied())
            .filter(|&c| c > 0)
            .map(|c| c as usize)
            .unwrap_or(16);
        Ok(Self {
            session,
            has_noise_input,
            hidden_channels,
        })
    }

    /// Channel count of the model's recurrent hidden state (read from
    /// the model at load). The `Vec` returned by [`Self::process_frame_luma`]
    /// has `hidden_channels() * width * height` elements.
    pub fn hidden_channels(&self) -> usize {
        self.hidden_channels
    }

    /// Process a luma field and return a restored version. Operates
    /// purely on Y (luma) data in `[0, 1]`.
    ///
    /// `noise_level` is the frame's link-quality conditioning in `[0, 1]`
    /// (1 = clean), fed to the model as a spatially-constant plane so it
    /// modulates denoising strength per frame. It MUST use the same
    /// normalisation the model was trained with — `CNR_FULL_SCALE_DB` in
    /// `models/train_temporal.py`: `(cnr_db / 30).clamp(0, 1)`. Callers
    /// should pass `estimate_cnr_db` through that map (see
    /// `FrameReconstructor::set_neural_noise_level`).
    pub fn process_frame_luma(
        &mut self,
        width: usize,
        height: usize,
        luma_in: &[f32],
        noise_level: f32,
        hidden_in: Option<&[f32]>,
        luma_out: &mut [f32],
    ) -> ort::Result<Vec<f32>> {
        assert_eq!(luma_in.len(), width * height);
        assert_eq!(luma_out.len(), width * height);

        let hidden_c = self.hidden_channels;
        let hidden_size = hidden_c * width * height;
        let mut default_hidden = Vec::new();
        let hidden_slice = if let Some(h) = hidden_in {
            assert_eq!(h.len(), hidden_size);
            h
        } else {
            default_hidden.resize(hidden_size, 0.0);
            &default_hidden
        };

        let input_tensor = Tensor::from_array(([1, 1, height, width], luma_in.to_vec()))?;
        let hidden_tensor =
            Tensor::from_array(([1, hidden_c, height, width], hidden_slice.to_vec()))?;

        // Feed the conditioning plane only to models that declare it, so
        // this stays compatible with pre-conditioning 2-input models.
        let outputs = if self.has_noise_input {
            let noise_tensor = Tensor::from_array((
                [1, 1, height, width],
                vec![noise_level.clamp(0.0, 1.0); width * height],
            ))?;
            self.session.run(inputs![
                "input" => input_tensor,
                "noise" => noise_tensor,
                "hidden_in" => hidden_tensor
            ])?
        } else {
            self.session
                .run(inputs!["input" => input_tensor, "hidden_in" => hidden_tensor])?
        };

        // Dims compared element-wise (no per-frame Vec allocation just
        // to shape-check).
        let dims_eq = |shape: &[i64], want: [usize; 4]| -> bool {
            shape.len() == 4 && shape.iter().zip(want).all(|(&s, w)| s as usize == w)
        };

        // Extract output image
        let (output_shape, output_data) = outputs["output"].try_extract_tensor::<f32>()?;
        if !dims_eq(output_shape, [1, 1, height, width]) {
            return Err(ort::Error::new("Output tensor shape mismatch".to_string()));
        }
        for (dst, &src) in luma_out.iter_mut().zip(output_data.iter()) {
            *dst = src.clamp(0.0, 1.0);
        }

        // Extract hidden out
        let (hidden_shape, hidden_data) = outputs["hidden_out"].try_extract_tensor::<f32>()?;
        if !dims_eq(hidden_shape, [1, hidden_c, height, width]) {
            return Err(ort::Error::new(
                "Hidden output tensor shape mismatch".to_string(),
            ));
        }

        Ok(hidden_data.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_neural_pass_through() {
        // Skip gracefully if the trained model isn't present (it ships
        // in `models/`, but a checkout without LFS/blobs might lack it).
        const MODEL: &str = "models/temporal_quantized_trained.onnx";
        if !Path::new(MODEL).exists() {
            return;
        }

        let mut restorer = NeuralRestorer::new(MODEL, false).expect("Failed to load model");
        let width = 640;
        let height = 480;

        let mut luma_in = vec![0.0f32; width * height];
        let mut luma_out = vec![0.0f32; width * height];
        luma_in[0] = 0.1;
        luma_in[1] = 0.9;
        luma_in[2] = 0.5;

        let expected_hidden = restorer.hidden_channels() * width * height;
        let hidden_out = restorer
            .process_frame_luma(width, height, &luma_in, 0.5, None, &mut luma_out)
            .unwrap();
        assert_eq!(hidden_out.len(), expected_hidden);

        for val in luma_out.iter() {
            assert!(*val >= 0.0 && *val <= 1.0);
        }
    }
}
