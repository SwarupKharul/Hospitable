from django.db import models
from django.contrib.auth.models import User
   
class Record(models.Model):
    objects = None
    title = models.CharField(max_length=100)
    created = models.DateTimeField(auto_now_add=True)
    upload = models.ImageField(upload_to ='records/') 
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return self.title