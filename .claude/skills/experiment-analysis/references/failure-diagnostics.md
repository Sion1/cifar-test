# Failure diagnostics — image classification

Common failure modes and what they look like in metrics + visualizations.
Use this when the headline metric moved unexpectedly, before declaring
**Bug** or **Failure**, and especially before iterating on the hypothesis.

The CIFAR-10 + ResNet-34 demo references are concrete; the patterns
generalize to any image-classification task.

---

## 1. Train-test leakage through augmentation

**Symptom.** Train accuracy and test accuracy track each other unusually
closely (gap < 1 pp). Test accuracy at epoch 1 is already ~0.5 instead of
~0.1 (random).

**Diagnosis.** Augmentation pipeline accidentally normalizes by per-batch
statistics, OR the test set was included in the training data, OR
`download=True` re-downloaded data into a wrong split.

**Verify.** Check `data.py`'s `build_transforms` — train and eval should
use **different** Compose pipelines. The eval pipeline must NOT include
random crop / flip / autoaugment. Re-run `evaluate(model, test_loader,
...)` after loading the pretrained checkpoint and confirm it matches the
report.

**Fix.** Separate transforms; verify with `print(len(trainset),
len(testset))` matches the dataset's published sizes.

---

## 2. Overfitting (the textbook case)

**Symptom.** Train acc keeps climbing past 0.99; test acc plateaus or
declines after some epoch.

**Diagnosis.** Capacity outsizes regularization. On CIFAR-10 with
ResNet-34 + 60 epochs, this commonly fires when augmentation is `none` or
weight decay is < 1e-4.

**Verify.** Plot per-epoch `train_acc` vs `test_acc` from
`runs/<exp>/history.json`. The "knee" where they diverge is the
overfitting onset. If the knee appears in the first 10 epochs, the
regularization is much too weak.

**Fix.** Strengthen augmentation (`standard → autoaugment`), increase
weight decay, add dropout, or shorten training. Don't conclude the model
"can't learn" without first ruling out overfitting.

---

## 3. Optimization not converging

**Symptom.** `train_acc` plateaus at chance (0.1 for CIFAR-10) or
oscillates wildly. `train_loss` flat or trending up.

**Diagnosis.** LR too high, gradient explosion, NaN in BatchNorm running
stats, or a typo in the loss function.

**Verify.** Print first-100-step loss values. If any is `nan` / `inf`, the
training silently crashed forward. If loss starts at ~`-log(1/10) = 2.3`
and stays flat, the model isn't learning at all.

**Fix.** Lower LR by 10×; check for `optim.step()` outside the gradient
loop; check `torch.isnan` on intermediate activations.

---

## 4. Class imbalance treated as if balanced

**Symptom.** Headline accuracy looks fine but per-class accuracy is
wildly uneven — some classes near 0%, others near 100%.

**Diagnosis.** The dataset is imbalanced and the loss is unweighted.
CIFAR-10 happens to be exactly balanced (5000/class), so this is more
common in custom datasets — but the same pattern appears if a sampler bug
causes one class to dominate batches.

**Verify.** Report per-class accuracy from the test set. Compute
`np.bincount(y_train)` to confirm the train distribution.

**Fix.** Class-weighted CE, oversample rare classes, or focal loss.

---

## 5. Augmentation breaks the label

**Symptom.** Test acc drops sharply after introducing a new augmentation,
even one that "should" help. Grad-CAM looks scrambled or focuses on
border pixels.

**Diagnosis.** The augmentation broke label correctness. CIFAR-10
examples: RandomCrop with too-large padding can crop the object out;
AutoAugment shears can clip key features; cutout placed at the center
destroys the class-defining region.

**Verify.** Save 16 augmented training samples to a grid PNG; eyeball
whether each is still recognizably its labeled class.

**Fix.** Tone down the augmentation strength or restrict to
label-preserving transforms.

---

## 6. Frozen-layer mistakes

**Symptom.** Train acc plateaus at a level too high for chance but too
low for "learning is working" (e.g. 0.4 on CIFAR-10). LR sweeps don't
move it.

**Diagnosis.** Some module's `requires_grad` got set to `False`
accidentally, e.g. when loading pretrained weights for transfer learning.

**Verify.** `print(sum(p.numel() for p in model.parameters() if
p.requires_grad))` — if this is much smaller than total params, you're
training a head-only linear probe by accident.

**Fix.** Re-enable grads; or if intentional, lower expectations
accordingly.

---

## 7. BatchNorm running-stats poisoning

**Symptom.** Single-image inference accuracy is much worse than batched
inference accuracy on the same data.

**Diagnosis.** BatchNorm's running mean/var got corrupted, or the model
wasn't put in `model.eval()` for inference.

**Verify.** Re-run evaluation explicitly with `model.eval()` and compare
to `model.train()` mode (the latter uses batch statistics).

**Fix.** Always `model.eval()` for inference; if the issue is corrupted
running stats, train one more epoch with the desired data distribution.

---

## 8. Spurious correlation / shortcut learning

**Symptom.** High accuracy but Grad-CAM consistently shows attention on
backgrounds or non-object regions. Per-class accuracy gap is wide between
classes whose backgrounds correlate with the label vs. those that don't.

**Diagnosis.** The model learned a shortcut (e.g. "ship images have water
backgrounds → predict ship from blue pixels"). On CIFAR-10 this is rarer
than on ImageNet but does occur for `ship` (water) and `airplane` (sky).

**Verify.** Run Grad-CAM on a balanced sample (8+ images per class).
Manually inspect whether attention sits on the object or the surroundings.

**Fix.** Background augmentation (random color jitter on background
pixels), mixup-style augmentations that decouple object from context, or
rebalance training data.

---

## 9. Optimizer / scheduler misconfiguration

**Symptom.** AdamW with lr=0.1 on a fresh model → diverges. SGD with
lr=0.001 → trains 10× too slow. Cosine scheduler with `T_max=epochs` but
training ran 2× the epochs → second half of training has lr ≈ 0.

**Diagnosis.** LR magnitude doesn't match optimizer family; or scheduler
horizon doesn't match actual training length.

**Verify.** Log `optimizer.param_groups[0]["lr"]` per epoch; plot it.

**Fix.** Standard ranges: SGD 0.01–0.2, AdamW 1e-4–5e-4 for ResNet-scale
models. Always set `T_max = epochs` for cosine.

---

## 10. Eval transform mismatch

**Symptom.** Saved checkpoint scores X% in training script's eval, but
`test.py` (or the dashboard's checkpoint loader) shows X−5%.

**Diagnosis.** Two evaluation paths apply different transforms —
typically one normalizes and the other doesn't, or one ToTensor()s in
float32 and the other keeps uint8.

**Verify.** Print the eval transform pipeline from both scripts side by
side. The CIFAR_MEAN / CIFAR_STD numbers must match exactly.

**Fix.** Centralize the transform definition in one module (`data.py`),
import it from both training and eval entry points.

---

## When to call **Bug** vs **Failure**

- **Bug** when one of the patterns above is the likely cause and the
  change itself wasn't supposed to break a sanity baseline. Fix the bug
  first; the verdict on the hypothesis comes after.
- **Failure** when the implementation is sound but the hypothesis didn't
  pan out — i.e. the predicted mechanism didn't fire and the metric moved
  in the wrong direction.

If you're not sure, default to **Bug** and investigate. A wrongly-labeled
**Bug** wastes one debug session; a wrongly-labeled **Failure** can
poison several follow-up iterations of the loop with bad lessons.
