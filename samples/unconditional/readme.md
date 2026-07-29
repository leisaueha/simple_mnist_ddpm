# unconditional

Uses the learned null label (-1). No class label is supplied, so guidance scale has no effect.

This case generates 36 images with seed `2026`.

```bash
python3 sample.py --checkpoint checkpoints/mnist_ddpm.pt --num-images 36 --out-dir samples/unconditional --seed 2026 --save-animation 
```
