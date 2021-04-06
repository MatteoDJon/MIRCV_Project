from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.core.files.storage import FileSystemStorage
from .utils import get_results






   
def index(request):
    uploaded_file_url=0
    if request.method == 'POST' and request.FILES['query_img']:
        myfile = request.FILES['query_img']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        
        results_url=get_results(filename)
        return render(request, 'faceup/index.html', {
            'uploaded_file_url': uploaded_file_url,
            'results_url': results_url
        })
    return render(request, 'faceup/index.html')
    


    

    
    



