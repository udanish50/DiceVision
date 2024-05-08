# DiceVision Perceptual Index (DVPI)

## Introduction
The DiceVision Perceptual Index (DVPI) introduces a novel metric for assessing the photorealistic image quality of AI-generated images by focusing on a high degree of alignment with human visual perception. Traditional metrics like FID and KID scores often diverge from human judgments. DVPI leverages advanced transformer-based attention mechanisms and Maximum Mean Discrepancy (MMD) to evaluate the perceptual integrity of generated images.

## Key Features
- **Advanced Evaluation Techniques**: Incorporates transformer-based attention mechanisms and MMD for nuanced perception analysis.
- **High Correlation with Human Judgment**: Demonstrates superior performance in aligning with human evaluative standards compared to traditional metrics such as FID, SSIM, and MS-SSIM.
- **Interpolative Binning Scale (IBS)**: Introduces a refined scaling method that enhances the interpretability of metric scores, making them more reflective of human assessments.

## Comparative Analysis
Comprehensive tests conducted across various generative models show that DVPI consistently outperforms existing metrics in terms of correlation with human judgments.

## Applications
DVPI provides a reliable tool for:
- Evaluating the photorealistic quality of AI-generated images.
- Assisting developers and researchers in enhancing image generation technologies.

## Future Work
The introduction of DVPI and IBS not only improves the reliability of assessments for AI-generated images but also opens pathways for future enhancements in image generation technologies.

## Acknowledgments
- Thanks to all contributors who have invested their time in improving the DVPI metric.
- Special thanks to [HCCG](https://thehccg.com/) for supporting this project.

## How to Use DVPI
To apply DVPI in your projects, follow these steps:
1. **Prepare Your Data**: Ensure your images are formatted correctly and stored in an accessible manner.
2. **Integrate DVPI Metrics**: Incorporate the DVPI code into your image processing pipeline. Detailed instructions and code snippets are provided in the `usage` section.
3. **Run Evaluations**: Execute your tests to generate DVPI scores and compare them with traditional metrics if needed.
