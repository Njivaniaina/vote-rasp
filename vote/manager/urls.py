from django.urls import path
from . import views

urlpatterns = [
    path('', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('reconnaissance/', views.reconnaissance_view, name='reconnaissance'),
    path('selection/', views.section_choice_view, name='selection'),
    path('vote/', views.cart_vote_view, name='vote'),
]

