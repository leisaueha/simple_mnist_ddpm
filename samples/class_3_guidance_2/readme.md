# class_3_guidance_2

Generates only MNIST class 3 with classifier-free guidance scale 2.

This case generates 36 images with seed `2026`.

```bash
python3 sample.py --checkpoint checkpoints/mnist_ddpm.pt --num-images 36 --out-dir samples/class_3_guidance_2 --seed 2026 --save-animation --labels 3 --guidance-scale 2 
```
