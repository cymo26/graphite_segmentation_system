import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import torch
import numpy as np
import os
import threading

# Import modulow
from segmentation import (
    MODELS_CONFIG, 
    predict_full_image
)
from analysis import analyze_graphite, FORM_LABELS, SIZE_LABELS
from mask_evaluation import evaluate_mask

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class GraphiteSegmentationApp:
    
    def __init__(self, root):
        self.root = root
        self.root.title("Segmentacja Grafitu - Mikrostruktura Zeliwa")
        self.root.geometry("1400x750")
        self.root.minsize(1200, 650)
        
        # Zmienne stanu
        self.current_image_path = None
        self.current_image_np = None
        self.current_mask = None
        self.loaded_models = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Sciezka do modeli i masek GT
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_path = os.path.join(self.script_dir, 'models')
        self.masks_gt_path = os.path.join(self.script_dir, '..', 'data', 'processed', 'masks')
        
        self._setup_ui()
        self._load_models()
    
    def _setup_ui(self):
        
        #Panel kontrolny
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)
        
        # Przycisk wczytania obrazu
        self.load_btn = ttk.Button(
            control_frame, 
            text="Wczytaj obraz", 
            command=self._load_image,
            width=20
        )
        self.load_btn.pack(side=tk.LEFT, padx=5)
    
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # Wybor modelu
        ttk.Label(control_frame, text="Model:").pack(side=tk.LEFT, padx=5)
        self.model_var = tk.StringVar(value=list(MODELS_CONFIG.keys())[0])
        self.model_combo = ttk.Combobox(
            control_frame, 
            textvariable=self.model_var,
            values=list(MODELS_CONFIG.keys()),
            state='readonly',
            width=25
        )
        self.model_combo.pack(side=tk.LEFT, padx=5)
        
        # Przycisk segmentacji
        self.segment_btn = ttk.Button(
            control_frame, 
            text="Segmentuj", 
            command=self._run_segmentation,
            width=15,
            state=tk.DISABLED
        )
        self.segment_btn.pack(side=tk.LEFT, padx=10)
        
        # Przycisk zapisu
        self.save_btn = ttk.Button(
            control_frame, 
            text="Zapisz maske", 
            command=self._save_mask,
            width=15,
            state=tk.DISABLED
        )
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        #SEKCJA ANALIZY
        ttk.Label(control_frame, text="Skala (um/px):").pack(side=tk.LEFT, padx=(5, 2))
        self.scale_var = tk.StringVar(value="0.5")
        self.scale_entry = ttk.Entry(control_frame, textvariable=self.scale_var, width=6)
        self.scale_entry.pack(side=tk.LEFT, padx=2)
        
        # Przycisk analizy
        self.analyze_btn = ttk.Button(
            control_frame, 
            text="Analiza ISO 945-1", 
            command=self._run_analysis,
            width=18,
            state=tk.DISABLED
        )
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        # Przycisk ewaluacji maski
        self.evaluate_btn = ttk.Button(
            control_frame, 
            text="Ewaluacja Maski", 
            command=self._open_evaluation_window,
            width=15,
            state=tk.DISABLED
        )
        self.evaluate_btn.pack(side=tk.LEFT, padx=5)
        
        # Info o urzadzeniu
        device_text = f"GPU: {torch.cuda.get_device_name(0)}" if self.device.type == 'cuda' else "CPU"
        ttk.Label(control_frame, text=f"[{device_text}]", foreground='green').pack(side=tk.RIGHT, padx=10)
        
        #Panel obrazow
        images_frame = ttk.Frame(self.root, padding="10")
        images_frame.pack(fill=tk.BOTH, expand=True)
        
        # Konfiguracja siatki
        images_frame.columnconfigure(0, weight=1)
        images_frame.columnconfigure(1, weight=1)
        images_frame.rowconfigure(0, weight=0)
        images_frame.rowconfigure(1, weight=1)
        
        # Etykiety
        ttk.Label(images_frame, text="Obraz wejsciowy", font=('Arial', 11, 'bold')).grid(
            row=0, column=0, pady=(0, 5)
        )
        ttk.Label(images_frame, text="Maska segmentacji", font=('Arial', 11, 'bold')).grid(
            row=0, column=1, pady=(0, 5)
        )
        
        # Canvas dla obrazu wejsciowego
        self.input_canvas = tk.Canvas(images_frame, bg='#2d2d2d', highlightthickness=1)
        self.input_canvas.grid(row=1, column=0, sticky='nsew', padx=(0, 5))
        
        # Canvas dla maski
        self.mask_canvas = tk.Canvas(images_frame, bg='#2d2d2d', highlightthickness=1)
        self.mask_canvas.grid(row=1, column=1, sticky='nsew', padx=(5, 0))
        
        # Bind resize
        self.input_canvas.bind('<Configure>', lambda e: self._display_images())
        self.mask_canvas.bind('<Configure>', lambda e: self._display_images())
        
        #Pasek statusu
        status_frame = ttk.Frame(self.root, padding="5")
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_var = tk.StringVar(value="Gotowy. Wczytaj obraz mikrostruktury.")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.LEFT)
        
        # Progress bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progress = ttk.Progressbar(
            status_frame, 
            mode='determinate', 
            length=200,
            variable=self.progress_var
        )
        self.progress.pack(side=tk.RIGHT, padx=10)
        
        # Label z liczba tile'ow
        self.tiles_var = tk.StringVar(value="")
        ttk.Label(status_frame, textvariable=self.tiles_var).pack(side=tk.RIGHT)
    
    def _load_models(self):
        self.status_var.set("Ladowanie modeli...")
        self.root.update()
        
        use_compile = hasattr(torch, 'compile') and self.device.type == 'cuda'
        
        for name, config in MODELS_CONFIG.items():
            model_path = os.path.join(self.models_path, config['folder'], config['file'])
            
            if os.path.exists(model_path):
                try:
                    model = config['class']()
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.to(self.device)
                    model.eval()
                    
                    # Kompilacja modelu
                    if use_compile:
                        try:
                            model = torch.compile(model, mode='reduce-overhead')
                            print(f"[OK] Zaladowano + skompilowano: {name}")
                        except Exception:
                            print(f"[OK] Zaladowano: {name} (bez kompilacji)")
                    else:
                        print(f"[OK] Zaladowano: {name}")
                    
                    self.loaded_models[name] = {
                        'model': model,
                        'tile_size': config['tile_size'],
                        'use_imagenet_norm': config['use_imagenet_norm'],
                        'iou': checkpoint.get('best_iou', None)
                    }
                except Exception as e:
                    print(f"[BLAD] Ladowanie {name}: {e}")
            else:
                print(f"[BRAK] Plik: {name} ({model_path})")
        
        if self.loaded_models:
            self.status_var.set(f"Zaladowano {len(self.loaded_models)} modeli. Wczytaj obraz.")
        else:
            self.status_var.set("UWAGA: Nie znaleziono zadnych modeli!")
            messagebox.showwarning(
                "Brak modeli",
                f"Nie znaleziono wytrenowanych modeli w:\n{self.models_path}\n\n"
                "Upewnij sie, ze pliki .pth istnieja w odpowiednich folderach."
            )
    
    def _load_image(self):
        filetypes = [
            ("Obrazy", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
            ("JPEG", "*.jpg *.jpeg"),
            ("PNG", "*.png"),
            ("Wszystkie pliki", "*.*")
        ]
        
        filepath = filedialog.askopenfilename(
            title="Wybierz obraz mikrostruktury",
            filetypes=filetypes
        )
        
        if filepath:
            try:
                img = Image.open(filepath).convert('RGB')
                self.current_image_np = np.array(img)
                self.current_image_path = filepath
                self.current_mask = None
                
                self._display_images()
                
                self.segment_btn.config(state=tk.NORMAL)
                self.save_btn.config(state=tk.DISABLED)
                
                h, w = self.current_image_np.shape[:2]
                filename = os.path.basename(filepath)
                self.status_var.set(f"Wczytano: {filename} ({w}x{h})")
                
            except Exception as e:
                messagebox.showerror("Blad", f"Nie mozna wczytac obrazu:\n{e}")
    
    def _display_images(self):
        
        # Obraz wejsciowy
        if self.current_image_np is not None:
            self._display_on_canvas(self.input_canvas, self.current_image_np)
        
        # Maska
        if self.current_mask is not None:
            # Konwertuj maske float do uint8
            mask_display = (self.current_mask * 255).astype(np.uint8)
            # Konwertuj do RGB
            mask_rgb = np.stack([mask_display] * 3, axis=-1)
            self._display_on_canvas(self.mask_canvas, mask_rgb)
    
    def _display_on_canvas(self, canvas, img_np):
        canvas.delete("all")
        
        canvas_w = canvas.winfo_width()
        canvas_h = canvas.winfo_height()
        
        if canvas_w < 10 or canvas_h < 10:
            return
        
        img_h, img_w = img_np.shape[:2]
        
        # Oblicz skale
        scale = min(canvas_w / img_w, canvas_h / img_h) * 0.95
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        
        # Resize
        img_pil = Image.fromarray(img_np)
        img_resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Konwertuj do PhotoImage
        photo = ImageTk.PhotoImage(img_resized)
        
        # Wycentruj na canvas
        x = (canvas_w - new_w) // 2
        y = (canvas_h - new_h) // 2
        
        canvas.create_image(x, y, anchor=tk.NW, image=photo)
        canvas.image = photo  # Zachowaj referencje
    
    def _run_segmentation(self):
        if self.current_image_np is None:
            messagebox.showwarning("Uwaga", "Najpierw wczytaj obraz!")
            return
        
        model_name = self.model_var.get()
        if model_name not in self.loaded_models:
            messagebox.showerror("Blad", f"Model '{model_name}' nie jest zaladowany!")
            return
        
        self.segment_btn.config(state=tk.DISABLED)
        self.load_btn.config(state=tk.DISABLED)
        self.progress_var.set(0)
        self.tiles_var.set("")
        self.status_var.set(f"Segmentacja: {model_name}...")
        
        thread = threading.Thread(target=self._segmentation_worker, args=(model_name,))
        thread.start()
    
    def _update_progress(self, current, total):
        if self.root.winfo_exists():
            percent = (current / total) * 100
            self.root.after(0, lambda: self.progress_var.set(percent))
            self.root.after(0, lambda: self.tiles_var.set(f"Tile {current}/{total}"))
    
    def _segmentation_worker(self, model_name):
        try:
            model_data = self.loaded_models[model_name]
            model = model_data['model']
            tile_size = model_data['tile_size']
            use_norm = model_data['use_imagenet_norm']
            
            # Segmentacja podzialem na tile'e
            pred = predict_full_image(
                model, 
                self.current_image_np, 
                self.device,
                tile_size=tile_size,
                overlap=tile_size // 4,
                use_imagenet_norm=use_norm,
                batch_size=16 if self.device.type == 'cuda' else 4,
                progress_callback=self._update_progress
            )
            
            # Binaryzacja
            self.current_mask = (pred > 0.5).astype(np.float32)
            
            # Aktualizuj UI w glownym watku
            if self.root.winfo_exists():
                self.root.after(0, self._segmentation_complete, model_name)
            
        except Exception as e:
            if self.root.winfo_exists():
                self.root.after(0, self._segmentation_error, str(e))
    
    def _segmentation_complete(self, model_name):
        self.progress_var.set(100)
        self.segment_btn.config(state=tk.NORMAL)
        self.load_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.NORMAL)
        self.analyze_btn.config(state=tk.NORMAL)
        self.evaluate_btn.config(state=tk.NORMAL)
        
        self._display_images()
        
        # Statystyki
        graphite_pct = self.current_mask.mean() * 100
        self.tiles_var.set("Gotowe")
        self.status_var.set(
            f"Segmentacja zakonczona ({model_name}) | "
            f"Udzial grafitu: {graphite_pct:.2f}%"
        )
    
    def _segmentation_error(self, error_msg):
        self.progress_var.set(0)
        self.tiles_var.set("")
        self.segment_btn.config(state=tk.NORMAL)
        self.load_btn.config(state=tk.NORMAL)
        self.status_var.set("Blad segmentacji!")
        messagebox.showerror("Blad segmentacji", error_msg)
    
    def _save_mask(self):
        if self.current_mask is None:
            messagebox.showwarning("Uwaga", "Brak maski do zapisania!")
            return
        
        #nazwa pliku
        if self.current_image_path:
            base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            default_name = f"{base_name}_mask.png"
        else:
            default_name = "mask.png"
        
        filepath = filedialog.asksaveasfilename(
            title="Zapisz maske",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("Wszystkie", "*.*")]
        )
        
        if filepath:
            try:
                mask_uint8 = (self.current_mask * 255).astype(np.uint8)
                Image.fromarray(mask_uint8).save(filepath)
                self.status_var.set(f"Zapisano maske: {os.path.basename(filepath)}")
            except Exception as e:
                messagebox.showerror("Blad", f"Nie mozna zapisac maski:\n{e}")
    
    def _run_analysis(self):
        if self.current_mask is None:
            messagebox.showwarning("Uwaga", "Najpierw wykonaj segmentacje!")
            return
        
        # Pobierz skale
        try:
            scale = float(self.scale_var.get())
            if scale <= 0:
                raise ValueError("Skala musi byc dodatnia")
        except ValueError as e:
            messagebox.showerror("Blad", f"Nieprawidlowa wartosc skali:\n{e}")
            return
        
        self.status_var.set("Analiza grafitu w toku...")
        self.analyze_btn.config(state=tk.DISABLED)
        self.root.update()
        
        # Uruchom analize w osobnym watku
        thread = threading.Thread(target=self._analysis_worker, args=(scale,))
        thread.start()
    
    def _show_analysis_results(self, results: dict):
        self.analyze_btn.config(state=tk.NORMAL)
        
        summary = results['summary']
        
        if summary['total_particles'] == 0:
            messagebox.showinfo("Analiza", "Nie znaleziono czastek grafitu w masce.")
            self.status_var.set("Analiza zakonczona - brak czastek")
            return
        
        # nowe okno z wynikami
        analysis_window = tk.Toplevel(self.root)
        analysis_window.title("Analiza grafitu - PN-EN ISO 945-1")
        analysis_window.geometry("1400x900")
        
        fig_frame = ttk.Frame(analysis_window)
        fig_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        if results['figure']:
            canvas = FigureCanvasTkAgg(results['figure'], master=fig_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Frame na przyciski
        btn_frame = ttk.Frame(analysis_window)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # eksportuj do CSV
        def export_csv():
            filepath = filedialog.asksaveasfilename(
                title="Eksportuj do CSV",
                defaultextension=".csv",
                filetypes=[("CSV", "*.csv")]
            )
            if filepath:
                summary['dataframe'].to_csv(filepath, index=False)
                messagebox.showinfo("Sukces", f"Zapisano: {filepath}")
        
        ttk.Button(btn_frame, text="Eksportuj CSV", command=export_csv).pack(side=tk.LEFT, padx=5)
        
        # Zapisz wizualizacje
        def save_figure():
            filepath = filedialog.asksaveasfilename(
                title="Zapisz wizualizacje",
                defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("PDF", "*.pdf")]
            )
            if filepath and results['figure']:
                results['figure'].savefig(filepath, dpi=150, bbox_inches='tight')
                messagebox.showinfo("Sukces", f"Zapisano: {filepath}")
        
        ttk.Button(btn_frame, text="Zapisz wykres", command=save_figure).pack(side=tk.LEFT, padx=5)
        
        # Zamknij
        ttk.Button(btn_frame, text="Zamknij", command=analysis_window.destroy).pack(side=tk.RIGHT, padx=5)
        
        # Aktualizuj status
        dominant_form = summary['dominant_form']
        dominant_size = summary['dominant_size_class']
        form_pct = summary['form_distribution'].get(dominant_form, 0)
        form_name = FORM_LABELS.get(dominant_form, dominant_form)
        size_label = SIZE_LABELS.get(dominant_size, str(dominant_size))
        
        self.status_var.set(
            f"Analiza: {summary['total_particles']} czastek | "
            f"Dominuje: {form_name} ({form_pct:.1f}%) | "
            f"Wielkosc: {size_label}"
        )
    
    def _analysis_error(self, error_msg: str):
        """Callback w przypadku bledu analizy"""
        self.analyze_btn.config(state=tk.NORMAL)
        self.status_var.set("Blad analizy!")
        messagebox.showerror("Blad analizy", error_msg)
    
    
    def _find_gt_mask_path(self):
        if self.current_image_path is None:
            return None
        
        # Sciezka do folderu raw
        raw_path = os.path.join(self.script_dir, '..', 'data', 'raw')
        raw_path = os.path.abspath(raw_path)
        
        # Sciezka do folderu masks
        masks_path = os.path.abspath(self.masks_gt_path)
        
        # Sprawdz czy obraz jest w folderze raw
        img_path = os.path.abspath(self.current_image_path)
        
        if not img_path.startswith(raw_path):
            # Obraz nie jest z folderu raw
            img_name = os.path.basename(img_path)
            name_base, ext = os.path.splitext(img_name)
            mask_name = f"{name_base}_mask{ext}"
            
            # Szukaj maski
            for root, dirs, files in os.walk(masks_path):
                if mask_name in files:
                    return os.path.join(root, mask_name)
            return None
        
        # Wyciagnij sciezke od raw
        rel_path = os.path.relpath(img_path, raw_path)
        
        # Zbuduj nazwe maski
        dir_part = os.path.dirname(rel_path)
        filename = os.path.basename(rel_path)
        name_base, ext = os.path.splitext(filename)
        mask_filename = f"{name_base}_mask{ext}"
        
        # Pelna sciezka do maski
        gt_mask_path = os.path.join(masks_path, dir_part, mask_filename)
        
        if os.path.exists(gt_mask_path):
            return gt_mask_path
        
        return None
    
    def _open_evaluation_window(self):
        if self.current_mask is None:
            messagebox.showwarning("Uwaga", "Najpierw wykonaj segmentacje!")
            return
        
        # nowe okno
        eval_window = tk.Toplevel(self.root)
        eval_window.title("Ewaluacja Maski - Porownanie z Ground Truth")
        eval_window.geometry("1400x900")
        
        # Zmienne 
        gt_mask_np = [None]
        results_data = [None]
        
        # Gorny panel kontrolny
        control_frame = ttk.Frame(eval_window, padding="10")
        control_frame.pack(fill=tk.X)
        
        ttk.Label(control_frame, text="Maska Ground Truth:").pack(side=tk.LEFT, padx=5)
        
        gt_path_var = tk.StringVar(value="(nie wczytano)")
        ttk.Label(control_frame, textvariable=gt_path_var, width=50).pack(side=tk.LEFT, padx=5)
        
        def load_gt_mask_from_path(filepath):
            if not filepath or not os.path.exists(filepath):
                return False
            try:
                gt_img = Image.open(filepath).convert('L')
                gt_mask_np[0] = np.array(gt_img)
                
                # wymiary
                pred_shape = self.current_mask.shape[:2]
                gt_shape = gt_mask_np[0].shape[:2]
                
                if pred_shape != gt_shape:
                    messagebox.showwarning(
                        "Rozne wymiary",
                        f"Maska predykcji: {pred_shape[1]}x{pred_shape[0]}\n"
                        f"Maska GT: {gt_shape[1]}x{gt_shape[0]}\n\n"
                        "Wymiary musza byc identyczne!"
                    )
                    gt_mask_np[0] = None
                    return False
                
                gt_path_var.set(os.path.basename(filepath))
                run_eval_btn.config(state=tk.NORMAL)
                return True
                
            except Exception as e:
                messagebox.showerror("Blad", f"Nie mozna wczytac maski:\n{e}")
                return False
        
        def load_gt_mask():
            filepath = filedialog.askopenfilename(
                title="Wybierz maske Ground Truth",
                initialdir=self.masks_gt_path,
                filetypes=[
                    ("Obrazy", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                    ("PNG", "*.png"),
                    ("Wszystkie", "*.*")
                ]
            )
            if filepath:
                load_gt_mask_from_path(filepath)
        
        ttk.Button(control_frame, text="Wczytaj GT", command=load_gt_mask).pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        def run_evaluation():
            if gt_mask_np[0] is None:
                messagebox.showwarning("Uwaga", "Najpierw wczytaj maske Ground Truth!")
                return
            
            # Konwertuj maske predykcji
            pred_mask = (self.current_mask * 255).astype(np.uint8)
            
            # Przeprowadz ewaluacje
            results = evaluate_mask(
                pred_mask=pred_mask,
                gt_mask=gt_mask_np[0],
                original_image=self.current_image_np
            )
            
            results_data[0] = results
            
            # Wyswietl wyniki
            display_results(results)
        
        run_eval_btn = ttk.Button(
            control_frame, 
            text="Uruchom ewaluacje", 
            command=run_evaluation,
            state=tk.DISABLED
        )
        run_eval_btn.pack(side=tk.LEFT, padx=5)
        
        def save_eval_figure():
            if results_data[0] is None or results_data[0]['figure'] is None:
                messagebox.showwarning("Uwaga", "Najpierw uruchom ewaluacje!")
                return
            
            filepath = filedialog.asksaveasfilename(
                title="Zapisz wizualizacje",
                defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("PDF", "*.pdf")]
            )
            if filepath:
                results_data[0]['figure'].savefig(filepath, dpi=150, bbox_inches='tight')
                messagebox.showinfo("Sukces", f"Zapisano: {filepath}")
        
        ttk.Button(control_frame, text="Zapisz wykres", command=save_eval_figure).pack(side=tk.LEFT, padx=5)
        
        # Model info
        model_name = self.model_var.get()
        ttk.Label(
            control_frame, 
            text=f"Model: {model_name}", 
            foreground='blue'
        ).pack(side=tk.RIGHT, padx=10)
        
        #Obszar na wyniki
        results_frame = ttk.Frame(eval_window, padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        placeholder_label = ttk.Label(
            results_frame, 
            text="Wczytaj maske Ground Truth i uruchom ewaluacje",
            font=('Arial', 14)
        )
        placeholder_label.pack(expand=True)
        
        canvas_widget = [None]
        
        def display_results(results):
            placeholder_label.pack_forget()
            
            # Usun poprzedni canvas jesli istnieje
            if canvas_widget[0]:
                canvas_widget[0].get_tk_widget().destroy()
            
            # Wyswietl figure
            if results['figure']:
                canvas = FigureCanvasTkAgg(results['figure'], master=results_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                canvas_widget[0] = canvas
            
            # Aktualizuj status glownego okna
            metrics = results['metrics']
            self.status_var.set(
                f"Ewaluacja: IoU={metrics['IoU']:.4f} | "
                f"F1={metrics['F1-Score']:.4f} | "
                f"Precision={metrics['Precision']:.4f} | "
                f"Recall={metrics['Recall']:.4f}"
            )
        
        # Przycisk zamkniecia
        ttk.Button(eval_window, text="Zamknij", command=eval_window.destroy).pack(pady=5)
        
        #Automatyczne wczytanie maski GT
        auto_gt_path = self._find_gt_mask_path()
        if auto_gt_path:
            if load_gt_mask_from_path(auto_gt_path):
                placeholder_label.config(
                    text="Maska GT wczytana automatycznie. Kliknij 'Uruchom ewaluacje'."
                )
