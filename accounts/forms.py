from django import forms
from django.contrib.auth.forms import UserCreationForm, UserChangeForm
from .models import CustomUser

class CustomUserCreationForm(UserCreationForm):
    first_name = forms.CharField(max_length=30, required=True, help_text="Required.")
    last_name = forms.CharField(max_length=30, required=True, help_text="Required.")

    class Meta:
        model = CustomUser
        fields = ['username', 'email', 'first_name', 'last_name', 'password1', 'password2']

class CustomUserChangeForm(UserChangeForm):
    class Meta:
        model = CustomUser
        fields = ['username', 'email', 'first_name', 'last_name']
