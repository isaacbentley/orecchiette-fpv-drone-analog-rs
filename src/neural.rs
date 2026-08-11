use ort::{inputs, session::Session, value::Tensor};

pub struct NeuralRestorer {
    session: Session,
}

impl NeuralRestorer {
    pub fn new(model_path: &str, use_gpu: bool) -> ort::Result<Self> {
        let mut builder = Session::builder()?;
        
        if use_gpu {
            let mut eps: Vec<ort::execution_providers::ExecutionProviderDispatch> = Vec::new();
            
            eps.push(ort::ep::CoreML::default().build());
            
            builder = builder.with_execution_providers(eps)?;
        }
        
        let num_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        let session = builder
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .with_intra_threads(num_cpus)?
            .with_inter_threads(num_cpus)?
            .commit_from_file(model_path)?;
        Ok(Self { session })
    }

    /// Process a luma field and return a restored version.
    /// Operates purely on Y (luma) data: 0.0 to 1.0.
    pub fn process_frame_luma(
        &mut self,
        width: usize,
        height: usize,
        luma_in: &[f32],
        hidden_in: Option<&[f32]>,
        luma_out: &mut [f32],
    ) -> ort::Result<Vec<f32>> {
        assert_eq!(luma_in.len(), width * height);
        assert_eq!(luma_out.len(), width * height);

        let hidden_size = 8 * width * height;
        let mut default_hidden = Vec::new();
        let hidden_slice = if let Some(h) = hidden_in {
            assert_eq!(h.len(), hidden_size);
            h
        } else {
            default_hidden.resize(hidden_size, 0.0);
            &default_hidden
        };

        let input_tensor = Tensor::from_array(([1, 1, height, width], luma_in.to_vec()))?;
        let hidden_tensor = Tensor::from_array(([1, 8, height, width], hidden_slice.to_vec()))?;
        
        let outputs = self.session.run(inputs!["input" => input_tensor, "hidden_in" => hidden_tensor])?;

        // Extract output image
        let (output_shape, output_data) = outputs["output"].try_extract_tensor::<f32>()?;
        if output_shape.iter().map(|&x| x as usize).collect::<Vec<_>>() != vec![1, 1, height, width] {
            return Err(ort::Error::new("Output tensor shape mismatch".to_string()));
        }

        // Copy back to output buffer
        for y in 0..height {
            for x in 0..width {
                luma_out[y * width + x] = output_data[y * width + x].max(0.0).min(1.0);
            }
        }

        // Extract hidden out
        let (hidden_shape, hidden_data) = outputs["hidden_out"].try_extract_tensor::<f32>()?;
        if hidden_shape.iter().map(|&x| x as usize).collect::<Vec<_>>() != vec![1, 8, height, width] {
            return Err(ort::Error::new("Hidden output tensor shape mismatch".to_string()));
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
        // Only run if the denoiser model has been generated
        if !std::path::Path::new("models/denoiser.onnx").exists() {
            return;
        }
        
        let mut restorer = NeuralRestorer::new("models/denoiser.onnx", false).expect("Failed to load model");
        let width = 640;
        let height = 480;
        
        let mut luma_in = vec![0.0f32; width * height];
        let mut luma_out = vec![0.0f32; width * height];
        luma_in[0] = 0.1;
        luma_in[1] = 0.9;
        luma_in[2] = 0.5;
        
        let hidden_out = restorer.process_frame_luma(width, height, &luma_in, None, &mut luma_out).unwrap();
        assert_eq!(hidden_out.len(), 8 * width * height);
        
        for val in luma_out.iter() {
            assert!(*val >= 0.0 && *val <= 1.0);
        }
    }
}
