# matplotlib/1 — the project's own issue report

- issue: https://github.com/matplotlib/matplotlib/issues/17542
- resolved: direct
- fetched: 2026-10-25T12:31:29Z
- fix-commit-date: 2020-06-09T02:43:42Z
- comments-included: 7
- gate-tree-sha256: b4a83b5d6c5665eb67d156d852e90c872c0384096f10f05201c0b81360121eda
- gate-answer-sha256: a861275a58045b0795266e8fea579f86ded869cd91832e3e4cffa7c7429659a6
- gate-removals: 0
- raw-sha256: 8f2b30199803d0574035fffc2b4201273d4ad4c7b01dc77f29672bf04fdc4ad7
- content-sha256: 8f2b30199803d0574035fffc2b4201273d4ad4c7b01dc77f29672bf04fdc4ad7
- raw-hints-sha256: c2bd94f9f8e17854f69f5c890120aac241f363cad3ea5de56eac83801b75b667
- hints-sha256: c2bd94f9f8e17854f69f5c890120aac241f363cad3ea5de56eac83801b75b667

<!-- gate -->

{
  "files": [
    "lib/matplotlib/tests/test_bbox_tight.py"
  ],
  "ok": true,
  "reason": null,
  "removals": [],
  "ungated_prose": [],
  "hints": {
    "removals": [],
    "ungated_prose": []
  }
}

<!-- raw -->

# Matplotlib 3.2.1 savefig empty image when fig size matches data size exactly

### Bug report

**Bug summary**

Saving a figure to an image file with the same pixel dimensions as data loaded with imshow does not work in 3.2.1, producing an empty image. If the dimensions are changed by even one pixel, normal behavior returns.

This bug appeared in 3.2.1 and I have confirmed that it does not occur with 3.1.1.

I noticed in the release notes for 3.2.0 that some changes were made to the default image interpolation; perhaps an unhandled condition was introduced here when no interpolation is required. However, I note that changing the interpolation mode had no effect on this bug.  

**Code for reproduction**

```python
import numpy as np
from matplotlib import pylab as plt

x_size, y_size = (100, 100)
fig = plt.figure(frameon=False, dpi=100, figsize=(y_size/100, x_size/100))
ax = plt.Axes(fig, [0., 0., 1., 1.])
fig.add_axes(ax)
ax.set_axis_off()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

data = np.random.rand(100, 100)
ax.imshow(data)
fig.savefig('/tmp/image.png', bbox_inches='tight', pad_inches=0)

# In matplotlib 3.2.1, empty file with size 100x100 pixels output
# if random data shape is changed to something other than 100x100, output
# works correctly.

# In matplotlib 3.1.1, file output is 100x100 with data shown correctly.
```

**Actual outcome**
![bug_image](https://user-images.githubusercontent.com/3060896/83362411-df247200-a345-11ea-9175-35959d8c5cf0.png)

**Expected outcome**

![working_image](https://user-images.githubusercontent.com/3060896/83362486-81dcf080-a346-11ea-89a8-e7c7f7b65373.png)

Produced with matplotlib 3.1.1

**Matplotlib version**
  * Operating system: Linux version 5.6.13-arch1-1
  * Matplotlib version: 3.2.1
  * Matplotlib backend (`print(matplotlib.get_backend())`): TkAgg
  * Python version: 3.8.3
  * Other libraries: numpy 1.18.4

Default system install on Arch linux package `python-matplotlib-3.2.1-1-x86_64`

<!-- content -->

# Matplotlib 3.2.1 savefig empty image when fig size matches data size exactly

### Bug report

**Bug summary**

Saving a figure to an image file with the same pixel dimensions as data loaded with imshow does not work in 3.2.1, producing an empty image. If the dimensions are changed by even one pixel, normal behavior returns.

This bug appeared in 3.2.1 and I have confirmed that it does not occur with 3.1.1.

I noticed in the release notes for 3.2.0 that some changes were made to the default image interpolation; perhaps an unhandled condition was introduced here when no interpolation is required. However, I note that changing the interpolation mode had no effect on this bug.  

**Code for reproduction**

```python
import numpy as np
from matplotlib import pylab as plt

x_size, y_size = (100, 100)
fig = plt.figure(frameon=False, dpi=100, figsize=(y_size/100, x_size/100))
ax = plt.Axes(fig, [0., 0., 1., 1.])
fig.add_axes(ax)
ax.set_axis_off()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

data = np.random.rand(100, 100)
ax.imshow(data)
fig.savefig('/tmp/image.png', bbox_inches='tight', pad_inches=0)

# In matplotlib 3.2.1, empty file with size 100x100 pixels output
# if random data shape is changed to something other than 100x100, output
# works correctly.

# In matplotlib 3.1.1, file output is 100x100 with data shown correctly.
```

**Actual outcome**
![bug_image](https://user-images.githubusercontent.com/3060896/83362411-df247200-a345-11ea-9175-35959d8c5cf0.png)

**Expected outcome**

![working_image](https://user-images.githubusercontent.com/3060896/83362486-81dcf080-a346-11ea-89a8-e7c7f7b65373.png)

Produced with matplotlib 3.1.1

**Matplotlib version**
  * Operating system: Linux version 5.6.13-arch1-1
  * Matplotlib version: 3.2.1
  * Matplotlib backend (`print(matplotlib.get_backend())`): TkAgg
  * Python version: 3.8.3
  * Other libraries: numpy 1.18.4

Default system install on Arch linux package `python-matplotlib-3.2.1-1-x86_64`

<!-- raw-hints -->

## Comments

### jklymak — 2020-05-31T21:25:33Z

If you remove `pad_inches=0` this works fine.  

This stopping working seems to bisect to a9b8b6e21d6ef1 which I think has more to do with extents than the image work.   https://github.com/matplotlib/matplotlib/pull/16734

### jklymak — 2020-05-31T21:33:23Z

Furthermore, if you do `pad_inches=0.0000000001` it works fine, or the axes size to 0.999999 etc it works fine.  So its definitely that you are hitting a precise integer somehow.  But if you don't do the tight_bbox, or pad_inches, you get exactly what you want.  

### rdgraham — 2020-05-31T22:33:54Z

Thanks for the quick response and work-around. Hopefully it will be possible to get to the underlying issue.

I don't recall the origin of the pad and bbox options in this code, though I would note that it is surprising that not specifying the pad_inches and bbox_inches options results in no padding in this case. From the documentation, the default value for pad_inches is 0.1 inch.

### jklymak — 2020-05-31T22:37:18Z

`pad_inches` only matters if you use `bbox_inches='tight'`

### tacaswell — 2020-06-03T18:09:04Z

This was broken by https://github.com/matplotlib/matplotlib/pull/16734 which disables the draw methods on the renderer (because to compute the bounding boxes we don't need to actually draw turn the crank on the renderer methods, only the artist methods) but we never re-set the methods to their original values.

This is not normally a problem as the tight bounding box is a different size that original size so the caching in https://github.com/matplotlib/matplotlib/blob/4356b677a33430b5c9e34e8b18e07cfd630a0704/lib/matplotlib/backends/backend_agg.py#L411-L421 misses and we get a new, un-patched renderer.  However, in this case the figure is already the correct size so we hit the cache and get back the renderer with the draw methods stripped off which then results in an empty output.

I can see a bunch of hacky ways to fix this, but not sure which one is the best.

### jklymak — 2020-06-03T18:51:26Z

It seems to me that `draw_disabled=True` should just make a new un-cached renderer?   

### tacaswell — 2020-06-03T19:36:57Z

The problem is that where we know if we have flagged it as `draw_disabled` (in backend bases) is not where we know if there is a cache or not (deep in the `print_XYZ` functions in the backends).

<!-- hints -->

## Comments

### jklymak — 2020-05-31T21:25:33Z

If you remove `pad_inches=0` this works fine.  

This stopping working seems to bisect to a9b8b6e21d6ef1 which I think has more to do with extents than the image work.   https://github.com/matplotlib/matplotlib/pull/16734

### jklymak — 2020-05-31T21:33:23Z

Furthermore, if you do `pad_inches=0.0000000001` it works fine, or the axes size to 0.999999 etc it works fine.  So its definitely that you are hitting a precise integer somehow.  But if you don't do the tight_bbox, or pad_inches, you get exactly what you want.  

### rdgraham — 2020-05-31T22:33:54Z

Thanks for the quick response and work-around. Hopefully it will be possible to get to the underlying issue.

I don't recall the origin of the pad and bbox options in this code, though I would note that it is surprising that not specifying the pad_inches and bbox_inches options results in no padding in this case. From the documentation, the default value for pad_inches is 0.1 inch.

### jklymak — 2020-05-31T22:37:18Z

`pad_inches` only matters if you use `bbox_inches='tight'`

### tacaswell — 2020-06-03T18:09:04Z

This was broken by https://github.com/matplotlib/matplotlib/pull/16734 which disables the draw methods on the renderer (because to compute the bounding boxes we don't need to actually draw turn the crank on the renderer methods, only the artist methods) but we never re-set the methods to their original values.

This is not normally a problem as the tight bounding box is a different size that original size so the caching in https://github.com/matplotlib/matplotlib/blob/4356b677a33430b5c9e34e8b18e07cfd630a0704/lib/matplotlib/backends/backend_agg.py#L411-L421 misses and we get a new, un-patched renderer.  However, in this case the figure is already the correct size so we hit the cache and get back the renderer with the draw methods stripped off which then results in an empty output.

I can see a bunch of hacky ways to fix this, but not sure which one is the best.

### jklymak — 2020-06-03T18:51:26Z

It seems to me that `draw_disabled=True` should just make a new un-cached renderer?   

### tacaswell — 2020-06-03T19:36:57Z

The problem is that where we know if we have flagged it as `draw_disabled` (in backend bases) is not where we know if there is a cache or not (deep in the `print_XYZ` functions in the backends).

