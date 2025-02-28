from django.urls import path
from .views import SignUpView, CustomLoginView, profile_view,home_view
from django.contrib.auth.views import LogoutView

urlpatterns = [
    path('signup/', SignUpView.as_view(), name='signup'),
    path('login/', CustomLoginView.as_view(), name='login'),
    path('logout/', LogoutView.as_view(), name='logout'),
    path('profile/', profile_view, name='profile'),
    path('', home_view, name='home'),  # Add home page
]
