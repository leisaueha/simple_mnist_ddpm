# all_classes_guidance_2

Cycles through labels 0-9 in one grid, making it easy to compare every class under the same sampling settings.

This case generates 36 images with seed `2026`.

```bash
python3 sample.py --checkpoint checkpoints/mnist_ddpm.pt --num-images 36 --out-dir samples/all_classes_guidance_2 --seed 2026 --save-animation --labels 0\,1\,2\,3\,4\,5\,6\,7\,8\,9 --guidance-scale 2 
```
