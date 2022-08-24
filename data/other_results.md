## GLUE results
We also evalute the language understanding performance of Uni-Perceiver on GLUE benchmarks. 
The results are listed as below.

<table border="1" width="100%">
  <tr align="center">
    <th>Dataset</th>
    <th>MNLI</th>
    <th>QNLI</th> <th>QQP</th> <th>RTE</th> <th>SST-2</th> <th>MRPC</th> <th>CoLA</th>
  </tr>

  <tr align="center">
    <td>Metric</td><td>Acc</td><td>Acc</td><td>F1</td><td>Acc</td><td>Acc</td><td>F1</td><td>Acc</td>
  </tr>


   <tr align="center">
    <td>Uni-Perceiver<sub>BASE</sub> </td><td>79.7</td><td>87.3</td><td>86.7 </td><td>71.1 </td><td>89.3 </td><td>86.0 </td><td>43.1</td> 
  </tr>
    <tr align="center">
    <td>Uni-Perceiver-MoE<sub>BASE</sub> </td><td>81.5</td><td>88.2</td><td>87.8 </td><td>75.8</td><td>90.9 </td><td>87.1 </td><td>52.2</td> 
  </tr>  
      <tr align="center">
    <td>Uni-Perceiver<sub>LARGE</sub> </td><td>82.5</td><td>89.2</td><td>87.7 </td><td>73.7</td><td>91.2 </td><td>90.2</td><td>52.0</td> 
  </tr> 
  <tr align="center">
    <td>Uni-Perceiver-MoE<sub>LARGE</sub> </td><td>85.7</td><td>91.9</td><td>89.5 </td><td>78.4</td><td>93.4 </td><td>91.2</td><td>57.4</td> 
  </tr>   
  </table>

  ---

* All fine-tuning experiments are performed on 1 GPU.

* We use the hyper-parameters for GLUE tasks from [fair-seq](https://github.com/facebookresearch/fairseq/blob/main/examples/bart/README.glue.md)

Model | MNLI | QNLI | QQP | RTE | SST-2 | MRPC | CoLA | STS-B
---|---|---|---|---|---|---|---|---
`--num-classes` | 3 | 2 | 2 | 2 | 2 | 2 | 2 | 1
`--lr` | 5e-6 | 1e-5 | 1e-5 | 1e-5 | 5e-6 | 2e-5 | 2e-5 | 2e-5
`bsz` | 128 | 32 | 32 | 32 | 128 | 64 | 64 | 32
`--total-num-update` | 30968 | 33112 | 113272 | 1018 | 5233 | 1148 | 1334 | 1799
`--warmup-updates` | 1858 | 1986 | 6796 | 61 | 314 | 68 | 80 | 107 | 1334 | 1799
`--warmup-updates` | 1858 | 1986 | 6796 | 61 | 314 | 68 | 80 | 107

* Following RoBerta, we finetune  RTE, STS and MRPC starting from the
MNLI single-task model, rather than the baseline
pretrained model.
