import numpy as np


def _bin(x, threshold=.5):
    return np.asarray(x).squeeze() >= threshold


def binary_metrics(pred, target, threshold=.5, eps=1e-8):
    p, y = _bin(pred, threshold), _bin(target, threshold)
    tp = float(np.logical_and(p, y).sum()); fp = float(np.logical_and(p, ~y).sum()); fn = float(np.logical_and(~p, y).sum())
    precision = tp / (tp + fp + eps); recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    return {"precision": precision, "recall": recall, "f1": f1, "dice": f1, "iou": iou}


def _skeleton(mask):
    """Small dependency-free Zhang-Suen thinning implementation."""
    img = _bin(mask).astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            remove = []
            for i in range(1, img.shape[0] - 1):
                for j in range(1, img.shape[1] - 1):
                    if img[i, j] == 0: continue
                    p = [img[i-1,j], img[i-1,j+1], img[i,j+1], img[i+1,j+1], img[i+1,j], img[i+1,j-1], img[i,j-1], img[i-1,j-1]]
                    transitions = sum(a == 0 and b == 1 for a, b in zip(p, p[1:] + p[:1]))
                    if sum(p) < 2 or sum(p) > 6 or transitions != 1: continue
                    if step == 0 and p[0]*p[2]*p[4] != 0: continue
                    if step == 0 and p[2]*p[4]*p[6] != 0: continue
                    if step == 1 and p[0]*p[2]*p[6] != 0: continue
                    if step == 1 and p[0]*p[4]*p[6] != 0: continue
                    remove.append((i, j))
            if remove:
                changed = True
                for i, j in remove: img[i, j] = 0
    return img.astype(bool)


def connected_components(mask):
    x = _bin(mask); seen = np.zeros_like(x, bool); count = 0
    for i, j in zip(*np.where(x)):
        if seen[i, j]: continue
        count += 1; stack = [(i, j)]; seen[i, j] = True
        while stack:
            a, b = stack.pop()
            for da, db in ((-1,0),(1,0),(0,-1),(0,1)):
                c, d = a + da, b + db
                if 0 <= c < x.shape[0] and 0 <= d < x.shape[1] and x[c,d] and not seen[c,d]:
                    seen[c,d] = True; stack.append((c,d))
    return count


def cldice(pred, target, threshold=.5, eps=1e-8):
    p, y = _bin(pred, threshold), _bin(target)
    sp, sy = _skeleton(p), _skeleton(y)
    tprec = (sp & y).sum() / (sp.sum() + eps); trec = (sy & p).sum() / (sy.sum() + eps)
    return float(2 * tprec * trec / (tprec + trec + eps))
