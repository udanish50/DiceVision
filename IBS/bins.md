# Classification Bins for Metrics

Below is a table categorizing different scores for various image quality metrics:

\[
\begin{array}{|c|c|c|c|c|c|c|c|c|}
\hline
\textbf{Criteria} & \textbf{Score} & \textbf{SSIM} & \textbf{PSNR} & \textbf{FID} & \textbf{MS-SSIM} & \textbf{LPIPS} & \textbf{DiceVision} & \textbf{Inception Score} \\
\hline
\text{Strongly Disagree} & 0.0 - 1.0 & -1 to -0.6 & 0-7 & >150 & -1 to -0.6 & 0.9 to 1 & >0.8 & 0 < 1 \\
\hline
\text{Somewhat Disagree} & 1.1 - 2.0 & -0.5 to -0.2 & 8-15 & 100 to 149 & -0.5 to -0.2 & 0.7 to 0.8 & 0.7 to 0.8 & 1 < 2 \\
\hline
\text{Neutral} & 2.1 - 3.0 & -0.1 to 0.2 & 16-23 & 31 to 99 & -0.1 to 0.2 & 0.5 to 0.6 & 0.5 to 0.6 & 2 < 3 \\
\hline
\text{Somewhat Agree} & 3.1 - 4.0 & 0.3 to 0.6 & 24-31 & 11 to 30 & 0.3 to 0.6 & 0.3 to 0.4 & 0.3 to 0.4 & 3 < 5 \\
\hline
\text{Strongly Agree} & 4.1 - 5.0 & 0.7 to 1.00 & >32 & <10 & 0.7 to 1.00 & 0 to 0.2 & 0 to 0.2 & >6 \\
\hline
\end{array}
\]

This table provides a structured overview of how each metric scales with perceived image quality across a standardized scoring system.
