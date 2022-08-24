# Checkpoints
We provide links for you to download the pre-trained weights of some Uni-Perceiver models. 


<table border="1" width="100%">
    <tr align="center">
        <th>Model</th><th>Link</th><th>Params</th><th>Hidden size</th><th>Intermediate size</th><th>Num. of heads</th><th>Enc layers</th>
    </tr>
    <tr align="center">
        <td>Uni Perceiver<sub>base </sub></td><td><a href="https://drive.google.com/file/d/1NjMrKWHGEigFO0SsLKPqCw1oz6IkKT7W/view?usp=sharing">Download<sup>torch</a> 
        <a href="https://drive.google.com/file/d/1aN1i9U56uON4ISwIamjfLTDCQBVzmFU3/view?usp=sharing">Download<sup>ds</a></td><td>124M</td><td>768</td><td>3072</td><td>12</td><td>12</td>
    </tr>
    <tr align="center">
        <td>Uni Perceiver MoE<sub>base</td><td><a href="https://drive.google.com/file/d/1w_CPXJGvE1XXpOpVDnxLBDFglrh-18S-/view?usp=sharing">Download<sup>ds</del></a></td><td>167M</td><td>768</td><td>3072</td><td>12</td><td>12</td>
    </tr>
    <tr align="center">
        <td>Uni Perceiver<sub>large</td><td><a href="https://drive.google.com/file/d/1G55-YmFIDCXJw3s98MtFB7yGtFjGIcXQ/view?usp=sharing">Download<sup>ds</a></td><td>354M</td><td>1024</td><td>4096</td><td>16</td><td>24</td>
    </tr>
    <tr align="center">
        <td>Uni Perceiver MoE<sub>large</td><td><a href="https://drive.google.com/file/d/1sj03SIKeVZpeGGFt-7srapO_abhu3ccz/view?usp=sharing">Download<sup>ds</a></td><td>505M</td><td>1024</td><td>4096</td><td>16</td><td>24</td>
    </tr>
    
</table>

* Models <sup>torch</sup> are pretrained in our current codebase, while  models  <sup>ds</sup> are pretrained in our previous codebase with <a href="https://github.com/microsoft/DeepSpeed">DeepSpeed </a> engine.

* `Params` in the table is the parameters required  during model deployment for image-text tasks. Note that it may be different on other tasks.


*  More pre-trained weights will be released.