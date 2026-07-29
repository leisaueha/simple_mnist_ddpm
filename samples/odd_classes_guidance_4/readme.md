# odd_classes_guidance_4

Cycles through labels 1, 3, 5, and 7. Scale 4 applies strong guidance, which may improve class fidelity while reducing variety. All guidance cases share the same seed for direct comparison.

This case generates 36 images with seed `2026`.

```bash
python3 sample.py --checkpoint checkpoints/mnist_ddpm.pt --num-images 36 --out-dir samples/odd_classes_guidance_4 --seed 2026 --save-animation --labels 1\,3\,5\,7 --guidance-scale 4 
```
