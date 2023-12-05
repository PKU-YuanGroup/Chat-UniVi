## Visualization
Place the CLIP model path into [Line 74](https://github.com/PKU-YuanGroup/Chat-UniVi/blob/57cba79acc7ce685c27eff40ed8f20fe1aee9a96/visualization.py#L74C1-L74C43) of ```visualization.py```.
```
clip_vit_14_path = ${openai_clip_path}
```

Then run:
```python
python visualization.py
```

You will get the following results:
<div align=center>
<img src="figures/input.jpg" width="100" />
<img src="figures/vanilla.jpg" width="100" />
<img src="figures/stage1.jpg" width="100" />
<img src="figures/stage2.jpg" width="100" />
<img src="figures/stage3.jpg" width="100" />
</div>
