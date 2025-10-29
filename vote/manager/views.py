import os
import base64
from django.conf import settings
from .forms import FaceCompareForm
import insightface
import cv2
import numpy as np
from .forms import LoginForm
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
import requests
from django.contrib.auth.models import User
from .forms import SectionChoiceForm


# --- Configuration model InsightFace (préchargé une seule fois) ---
# Utiliser CPU par défaut (-1). Si tu veux GPU, mets ctx_id=0 et assure-toi d'avoir le GPU configuré.
model = insightface.app.FaceAnalysis(name="buffalo_l")
model.prepare(ctx_id=-1, det_size=(640, 640))

# --- Dossier uploads (préchargement au démarrage) ---
UPLOADS_SUBDIR = "uploads"
uploads_dir = os.path.join(getattr(settings, "MEDIA_ROOT", "media"), UPLOADS_SUBDIR)

# S'assurer que le dossier existe
os.makedirs(uploads_dir, exist_ok=True)

# Liste préchargée : chaque élément = {'filename': ..., 'embedding': np.array, 'img_b64': 'data:image/jpeg;base64,...'}
preloaded_database = []

def _img_to_base64(img):
    """Encode image OpenCV (BGR) en base64 pour afficher dans un <img src="data:...">."""
    ret, buf = cv2.imencode('.jpg', img)
    if not ret:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"

def _load_image_cv2(path):
    """Charger une image depuis un chemin (en BGR)."""
    img = cv2.imread(path)
    return img

def _compute_face_embedding(img):
    """Retourne l'embedding du premier visage détecté ou None si aucun visage."""
    if img is None:
        return None
    faces = model.get(img)
    if not faces:
        return None
    emb = faces[0].embedding
    return emb

# Précharger toutes les images du dossier uploads au démarrage
print(f"[INFO] Chargement des images depuis : {uploads_dir}")

for fname in sorted(os.listdir(uploads_dir)):
    full = os.path.join(uploads_dir, fname)
    if not os.path.isfile(full):
        continue
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        continue

    print(f"[LOAD] Lecture de {fname} ...")

    try:
        img = _load_image_cv2(full)
        emb = _compute_face_embedding(img)
        if emb is None:
            print(f"[WARN] Aucun visage détecté dans {fname}")
            continue
        img_b64 = _img_to_base64(img) or ""
        preloaded_database.append({
            'filename': fname,
            'path': full,
            'embedding': emb,
            'img_b64': img_b64,
        })
        print(f"[OK] Image '{fname}' chargée avec succès ✅")

    except Exception as e:
        print(f"[ERROR] Échec de chargement de {fname} : {e}")
        continue

print(f"[INFO] Total d’images chargées : {len(preloaded_database)}")


# --- Vue Django ---
def reconnaissance_view(request):
    """
    Vue qui prend une image uploadée via le template (form),
    compare son embedding avec les embeddings préchargés dans uploads/,
    et renvoie le meilleur match et la similarité.
    """

    if not request.user.is_authenticated:
        # Rediriger vers la page de login si non connecté
        return redirect('/')

    print(request.session['choix'])


    similarity = None
    message = None
    best_match = None
    uploaded_img_b64 = None

    if request.method == 'POST':
        form = FaceCompareForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['image1']  # on suppose 'image1' dans le form
            # Lire l'image uploadée en mémoire (cv2.imdecode)
            try:
                file_bytes = uploaded_file.read()
                nparr = np.frombuffer(file_bytes, np.uint8)
                img1 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:
                img1 = None

            if img1 is None:
                message = "Impossible de lire l'image uploadée."
            else:
                # encoder pour affichage dans template
                uploaded_img_b64 = _img_to_base64(img1)

                emb1 = _compute_face_embedding(img1)
                if emb1 is None:
                    message = "Aucun visage détecté dans l'image uploadée."
                elif not preloaded_database:
                    message = "Aucune image dans le dossier 'uploads' pour comparer."
                else:
                    # comparer avec chaque embedding préchargé
                    best_sim = -1.0
                    best_item = None
                    for item in preloaded_database:
                        emb2 = item.get('embedding')
                        if emb2 is None:
                            continue
                        # Similarité cosinus
                        sim = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
                        if sim > best_sim:
                            best_sim = sim
                            best_item = item

                    if best_item is None:
                        message = "Pas de visages valides dans la base 'uploads'."
                    else:
                        similarity = best_sim
                        best_match = {
                            'filename': best_item['filename'],
                            'img_b64': best_item['img_b64'],
                        }
                        # seuil — tu peux l'ajuster
                        seuil = 0.4
                        if similarity > seuil:
                            message = f"Même personne (similarité={similarity:.3f} filename {best_match['filename']})"
                        else:
                            message = f"Personnes différentes (similarité={similarity:.3f})"
    else:
        form = FaceCompareForm()

    return render(request, 'manager/reconnaissance.html', {
        'form': form,
        'similarity': similarity,
        'message': message,
        'best_match': best_match,         # dict avec 'filename' et 'img_b64' (ou None)
        'uploaded_img_b64': uploaded_img_b64,  # data url de l'image uploadée
        'preloaded_count': len(preloaded_database),
    })



def login_view(request):
    """
    Vue pour la connexion des utilisateurs.
    """

    if request.user.is_authenticated:
        return redirect('selection')

    message = None

    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            matricule = form.cleaned_data['matricule']
            password = form.cleaned_data['password']

             # 🔹 Vérification auprès de la base externe
            try:
                response = requests.post(
                    'http://192.168.11.116:4000/auth/login',
                    json={'matricule': matricule, 'password': password},
                    timeout=5
                )

                data = response.json()
                if data.get('msg'):
                    print(data.get('msg'))
                    message = "❌ Utilisateur inexistant dans la base externe"
                    
                else:
                    message = "✅ Utilisateur vérifié sur la base externe"
                    print("✅ Utilisateur vérifié sur la base externe")
                    print(data)

                    # 🔹 Optionnel : créer l'utilisateur local s'il n'existe pas
                    user, created = User.objects.get_or_create(username=matricule)
                    if created:
                        user.set_password(password)
                        user.save()

                    # 🔹 Connexion locale Django
                    user = authenticate(request, username=matricule, password=password)
                    if user:
                        login(request, user)
                        return redirect('selection')

            except requests.RequestException as e:
                print("Erreur lors de la requête externe :", e)
                message = "❌ Impossible de contacter la base externe"

            
    else:
        form = LoginForm()

    return render(request, 'manager/login.html', {'form': form, 'message': message})

def logout_view(request):
    """
    Déconnexion utilisateur.
    """
    request.session.pop('choix', None)
    logout(request)
    return redirect('/')


def section_choice_view(request):
    
    if not request.user.is_authenticated:
        return redirect('/')

    if 'choix' in request.session:
        return redirect('reconnaissance')  # par exemple

    # 🔹 Récupérer la liste des sections depuis l'API
    try:
        response = requests.get("http://192.168.11.116:4000/election/electionavailable", timeout=5)
        response.raise_for_status()
        data = response.json()  # supposons: [{"id":1, "name":"Section A"}, ...]
        
        sections_choices = [(str(item['id']), item['name']) for item in data]
    except requests.RequestException as e:
        print("Erreur API:", e)
        sections_choices = []

    # 🔹 Initialiser le formulaire avec ces choix
    form = SectionChoiceForm()
    form.fields['choix'].choices = sections_choices

    choix = None  # variable pour stocker la sélection

    if request.method == 'POST':
        form = SectionChoiceForm(request.POST)
        form.fields['choix'].choices = sections_choices  # obligatoire à refaire lors du POST
        if form.is_valid():
            choix = form.cleaned_data['choix']

            # récupérer toutes les infos de l'item sélectionné
            item = next((x for x in data if str(x['id']) == choix), None)
            if item:
                # 🔹 stocker toutes les variables dans la session
                request.session['choix'] = item
                return redirect('reconnaissance')  # rediriger après choix

    return render(request, 'manager/selection.html', {
        'form': form,
        'choix': choix
    })

