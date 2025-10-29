from django import forms

class FaceCompareForm(forms.Form):
    image1 = forms.ImageField(label="Choisissez une image à comparer")


class LoginForm(forms.Form):
    matricule = forms.CharField(
        label="Matricule",
        max_length=150,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Votre identifiant',
        })
    )
    password = forms.CharField(
        label="Mot de passe",
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Votre mot de passe',
        })
    )

class SectionChoiceForm(forms.Form):
    choix = forms.ChoiceField(label="Choisir une section", choices=[])

