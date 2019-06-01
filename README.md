<div class="container">
  <div class="row">
    <div class="col-sm">
    </div>
    <div class="col">
      <h1 align="justify">
        Semi-supervised learning with Semantic Loss
      </h1>
    </div>
  </div>
</div>

<h3> Results </h3>

<h4> Isotropic Gaussian blobs </h4>

<h4> Fashion MNIST </h4>

| Model    | Labeled | Unlabeled | CE Loss | Sem Loss (lab)  | Sem Loss (unlab) | Accuracy |
| -------- | :-----: | :-------: | :-----: | :-------------: | :--------------: | -------: |
| Baseline | 120     | 0         | Yes     | No              | No               | 0.6204   |
|          | 120     | 0         | Yes     | Yes (w_s = 0.5) | No               | 0.6716   |
|          | 120     | All       | Yes     | No              | Yes (annealed)   |          |
|          | 120     | All       | Yes     | Yes (annealed)  | Yes (annealed)   |          |

<h4> Reuters </h4>


<h3> References </h3>

<ol>
  <li align="justify"> Xu et al. (2018). A Semantic Loss Function for Deep Learning with Symbolic Knowledge. <i> arXiv preprint arXiv:1711.11157 </i>.
</ol>


## Install 

```
pip3 install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
pip3 install torchvision
pip install -r requirements.txt
python -m spacy download en

mkdir logs
```
