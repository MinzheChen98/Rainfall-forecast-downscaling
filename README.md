# Rainfall-forecast-downscaling
I am working on this project now

Input is low resolution (LR) climate images, including precipitation, temperature and so on.
Model I am using now is VDSR
Output is super resolution (SR) rainfall image

Input includes 11 ensemble members (11 forecast image), each has error compared with ground truth value.
I need to enhance the resolution, and reduce the error to make the forecast more accurate, this is the reason why other climate images like temperature are needed.

The output is evaluated by CRPS score, it shows that my model is better than traditional methods.
Some input, output and label images are shown here, we can see that error of output image is much less than input

Some (mainly data preprocessing tool and evaluation) are adapted from Weifan Jiang, https://github.com/JiangWeiFanAI/iu60_csiro
