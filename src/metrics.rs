//! Fidelity metrics for reconstructed video frames.

/// Computes Peak Signal-to-Noise Ratio (PSNR) between a test frame and a ground truth frame.
/// Both frames should be packed `u32` (e.g. ARGB/XRGB). We extract the lower 8 bits (Luma) for comparison.
pub fn compute_psnr(test: &[u32], truth: &[u32]) -> f64 {
    assert_eq!(test.len(), truth.len());
    let mut mse = 0.0f64;
    for (a, b) in test.iter().zip(truth.iter()) {
        let diff = (a & 0xFF) as f64 - (b & 0xFF) as f64;
        mse += diff * diff;
    }
    mse /= test.len() as f64;
    if mse < 1e-10 {
        return f64::INFINITY; // Perfect match
    }
    10.0 * (255.0f64 * 255.0f64 / mse).log10()
}

/// Computes the Pearson correlation coefficient of spatial gradients (Sobel-like) between two frames.
/// The frames must have the given `width` and `height`.
pub fn compute_gradient_correlation(test: &[u32], truth: &[u32], width: usize, height: usize) -> f64 {
    assert_eq!(test.len(), width * height);
    assert_eq!(truth.len(), width * height);
    
    // We compute simple forward differences (dx, dy) to get the gradient magnitude.
    let mut mean_test = 0.0f64;
    let mut mean_truth = 0.0f64;
    
    let mut grads_test = Vec::with_capacity((width - 1) * (height - 1));
    let mut grads_truth = Vec::with_capacity((width - 1) * (height - 1));
    
    for y in 0..(height - 1) {
        for x in 0..(width - 1) {
            let idx = y * width + x;
            
            let t_xy = (test[idx] & 0xFF) as f64;
            let t_x1y = (test[idx + 1] & 0xFF) as f64;
            let t_xy1 = (test[idx + width] & 0xFF) as f64;
            let grad_test = ((t_x1y - t_xy).powi(2) + (t_xy1 - t_xy).powi(2)).sqrt();
            grads_test.push(grad_test);
            mean_test += grad_test;
            
            let r_xy = (truth[idx] & 0xFF) as f64;
            let r_x1y = (truth[idx + 1] & 0xFF) as f64;
            let r_xy1 = (truth[idx + width] & 0xFF) as f64;
            let grad_truth = ((r_x1y - r_xy).powi(2) + (r_xy1 - r_xy).powi(2)).sqrt();
            grads_truth.push(grad_truth);
            mean_truth += grad_truth;
        }
    }
    
    let n = grads_test.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    mean_test /= n;
    mean_truth /= n;
    
    let mut cov = 0.0f64;
    let mut var_test = 0.0f64;
    let mut var_truth = 0.0f64;
    
    for (gt, gr) in grads_test.iter().zip(grads_truth.iter()) {
        let dt = gt - mean_test;
        let dr = gr - mean_truth;
        cov += dt * dr;
        var_test += dt * dt;
        var_truth += dr * dr;
    }
    
    if var_test == 0.0 || var_truth == 0.0 {
        return 0.0;
    }
    
    cov / (var_test * var_truth).sqrt()
}
