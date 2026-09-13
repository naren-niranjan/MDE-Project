# Internal tensor shape against processing resolution

Source image 2448 x 2048 (WxH). Shapes are what `input_processor` returns, which is what the network is handed.

## `upper_bound_resize`

| requested | full shape | H x W | longer side | linear to source | area | tracks? |
|---|---|---|---|---|---|---|
| 252 | 1x3x210x252 | 210 x 252 | 252 | 9.71 | 94.4 | yes |
| 504 | 1x3x420x504 | 420 x 504 | 504 | 4.86 | 23.6 | yes |
| 700 | 1x3x588x700 | 588 x 700 | 700 | 3.50 | 12.2 | yes |
| 1008 | 1x3x840x1008 | 840 x 1008 | 1008 | 2.43 | 5.9 | yes |
| 1400 | 1x3x1176x1400 | 1176 x 1400 | 1400 | 1.75 | 3.1 | yes |
| 2002 | 1x3x1680x2002 | 1680 x 2002 | 2002 | 1.22 | 1.5 | yes |
| 2450 | 1x3x2044x2450 | 2044 x 2450 | 2450 | 1.00 | 1.0 | yes |

Tracks the parameter at: 252, 504, 700, 1008, 1400, 2002, 2450
Does not track at: none

## LaTeX table for Section 5.8

\begin{table}[H]
  \centering
  \caption{Tensor shape handed to the network against the
           requested processing resolution, read from the
           model's own input processor. The upsampling
           factor back to the source follows from it.}
  \label{tab:meth_tensor_shapes}
  \small
  \begin{tabularx}{\textwidth}{@{}Xlrr@{}}
    \toprule
    \thead{Requested (px)} & \thead{Tensor $H \times W$} & \thead{Linear} & \thead{Area} \\
    \midrule
    252 & $210 \times 252$ & 9.71 & 94.4 \\
    504 & $420 \times 504$ & 4.86 & 23.6 \\
    700 & $588 \times 700$ & 3.50 & 12.2 \\
    1008 & $840 \times 1008$ & 2.43 & 5.9 \\
    1400 & $1176 \times 1400$ & 1.75 & 3.1 \\
    2002 & $1680 \times 2002$ & 1.22 & 1.5 \\
    2450 & $2044 \times 2450$ & 1.00 & 1.0 \\
    \bottomrule
  \end{tabularx}
\end{table}
