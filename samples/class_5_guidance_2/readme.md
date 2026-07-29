# class_5_guidance_2

Generates only MNIST class 5 with classifier-free guidance scale 2.

This case generates 36 images with seed `2026`.

```bash
python3 sample.py --checkpoint checkpoints/mnist_ddpm.pt --num-images 36 --out-dir samples/class_5_guidance_2 --seed 2026 --save-animation --labels 5 --guidance-scale 2 
```
