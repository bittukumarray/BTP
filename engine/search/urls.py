from django.urls import path, include
from .views import *


urlpatterns = [
    path('get-docs/', get_ranked_docs.as_view()),
]