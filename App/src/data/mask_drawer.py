import os
import json
import cv2
import numpy as np

def find_label_jsons(root_dir):
	for dirpath, _, filenames in os.walk(root_dir):
		for filename in filenames:
			if filename.endswith('__labels.json'):
				yield os.path.join(dirpath, filename)

def draw_mask_from_json(json_path, image_shape):
	with open(json_path, 'r', encoding='utf-8') as f:
		data = json.load(f)
	mask = np.zeros(image_shape[:2], dtype=np.uint8)
	labels = data.get('labels', [])
	for label in labels:
		for region in label.get('regions', []):
			pts = np.array([[p['x'], p['y']] for p in region], dtype=np.int32)
			if len(pts) > 2:
				cv2.fillPoly(mask, [pts], color=255)
	return mask

def main():
	root_dir = os.path.join(os.path.dirname(__file__), 'data')
	for json_path in find_label_jsons(root_dir):
		with open(json_path, 'r', encoding='utf-8') as f:
			data = json.load(f)
		img_filename = data.get('image_filename')
		if not img_filename:
			print(f'Brak image_filename w {json_path}')
			continue
		img_path = os.path.join(os.path.dirname(json_path), img_filename)
		if not os.path.exists(img_path):
			# Spróbuj bez :Zone.Identifier
			img_path = img_path.split(':')[0]
			if not os.path.exists(img_path):
				print(f'Brak obrazu: {img_path}')
				continue
		image = cv2.imread(img_path)
		if image is None:
			print(f'Nie można wczytać obrazu: {img_path}')
			continue
		mask = draw_mask_from_json(json_path, image.shape)
		base = os.path.splitext(img_filename)[0]
		mask_filename = os.path.join(os.path.dirname(json_path), base + '_mask.jpg')
		cv2.imwrite(mask_filename, mask)
		print(f'Zapisano maskę: {mask_filename}')

if __name__ == '__main__':
	main()
