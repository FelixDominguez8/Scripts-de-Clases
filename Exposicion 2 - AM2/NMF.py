import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import NMF
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d


def pasar_a_grayscale(path, max_size=256):
	imagen_con = Image.open(path).convert("L")
	w, h = imagen_con.size
	escala = 1.0
	if max(w, h) > max_size:
		escala = max_size / float(max(w, h))
		imagen_con = imagen_con.resize((int(w * escala), int(h * escala)), Image.LANCZOS)
	arr = np.asarray(imagen_con, dtype=float) / 255.0
	return arr


def matriz_patches(img, patch_size=16, max_patches=2000, random_state=0):
	h, w = img.shape
	p = patch_size
	if h < p or w < p:
		p = min(h, w)
	pad_h = (p - (h % p)) % p
	pad_w = (p - (w % p)) % p
	img_p = np.pad(img, ((0, pad_h), (0, pad_w)), mode="reflect")
	Hp, Wp = img_p.shape
	bh = Hp // p
	bw = Wp // p
	patches = img_p.reshape(bh, p, bw, p).transpose(0, 2, 1, 3).reshape(bh * bw, p, p)
	n_patches = patches.shape[0]
	X = patches.reshape(n_patches, -1)
	return X, patches, (h, w), p, (pad_h, pad_w), (bh, bw)


def para_nmf(X, n_components=16, max_iter=200, random_state=0):
	model = NMF(n_components=n_components, init="nndsvda", max_iter=max_iter, random_state=random_state)
	W = model.fit_transform(X)
	H = model.components_
	return model, W, H


def patches_componentes(H, patch_size):
	componentes = H.reshape((-1, patch_size, patch_size))
	componentes_n = []
	for c in componentes:
		mn = c.min()
		mx = c.max()
		if mx - mn > 1e-12:
			componentes_n.append((c - mn) / (mx - mn))
		else:
			componentes_n.append(np.zeros_like(c))
	return np.array(componentes_n)


def reconstruir(patches, shape_imagen, pad, blocks):
	pad_h, pad_w = pad
	bh, bw = blocks
	p = patches.shape[1]
	grid = patches.reshape(bh, bw, p, p).transpose(0, 2, 1, 3).reshape(bh * p, bw * p)
	H, W = shape_imagen
	return grid[:H, :W]


def matriz_importancia(W, patches, shape_imagen, patch_size, pad, blocks):
	n_patches = W.shape[0]
	n_components = W.shape[1]
	maps = []
	for k in range(n_components):
		weight_patches = (W[:, k][:, None, None] * np.ones_like(patches))
		imp_map = reconstruir(weight_patches, shape_imagen, pad, blocks)
		maps.append(imp_map)
	return np.array(maps)


def reconstruir_imagen(W, H, patches, shape_imagen, pad, blocks):
	p = patches.shape[1]
	patches_reconstruidos = (W @ H).reshape((-1, p, p))
	imagen_reconstruida = reconstruir(patches_reconstruidos, shape_imagen, pad, blocks)
	return imagen_reconstruida


def para_rasgos(images, titles, cols=4, cmap="gray", figsize=(10, 8), vmin=None, vmax=None):
	n = len(images)
	rows = math.ceil(n / cols)
	fig, axs = plt.subplots(rows, cols, figsize=figsize)
	axs = axs.flatten()
	for i in range(rows * cols):
		axs[i].axis("off")
	for i, img in enumerate(images):
		axs[i].imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
		if i < len(titles):
			axs[i].set_title(titles[i], fontsize=8)
	plt.tight_layout()
	return fig


def main():
	imagen = "sho.jpeg"
	img = pasar_a_grayscale(imagen, max_size=256)
	X, patches, shape_imagen, patch_size, pad, blocks = matriz_patches(
		img, patch_size=16, max_patches=2000, random_state=0
	)

	model, W, H = para_nmf(X, n_components=16, max_iter=200, random_state=0)

	componentes = patches_componentes(H, patch_size)

	imp_maps = matriz_importancia(W, patches, shape_imagen, patch_size, pad, blocks)

	imagen_reconstruida = reconstruir_imagen(W, H, patches, shape_imagen, pad, blocks)

	fig = plt.figure(figsize=(12, 9))
	gs = fig.add_gridspec(2, 2)

	ax0 = fig.add_subplot(gs[0, 0])
	ax0.imshow(img, cmap="gray")
	ax0.set_title("Original")
	ax0.axis("off")

	n = componentes.shape[0]
	comp_grid = []
	comp_titles = []
	for i in range(n):
		comp_grid.append(componentes[i])
		comp_titles.append(f"Comp {i}")
	fig1 = para_rasgos(comp_grid, comp_titles, cols=min(n, 8), cmap="gray", figsize=(8, 6))
	fig1.suptitle("Matriz de rasgos")

	imp_grid = []
	imp_titles = []
	for i in range(imp_maps.shape[0]):
		imap = imp_maps[i]
		mn = imap.min()
		mx = imap.max()
		if mx - mn > 1e-12:
			imp_grid.append((imap - mn) / (mx - mn))
		else:
			imp_grid.append(np.zeros_like(imap))
		imp_titles.append(f"Imp {i}")
	fig2 = para_rasgos(imp_grid, imp_titles, cols=min(imp_maps.shape[0], 8), cmap="inferno", figsize=(8, 6))
	fig2.suptitle("Matriz de importancia por rasgo")

	ax3 = fig.add_subplot(gs[0, 1])
	ax3.imshow(imagen_reconstruida, cmap="gray")
	ax3.set_title("Aproximacion")
	ax3.axis("off")

	plt.show()


if __name__ == "__main__":
	main()