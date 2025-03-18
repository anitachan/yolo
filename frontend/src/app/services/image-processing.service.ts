import { HttpClient } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ImageProcessingService {

  private readonly apiUrl = 'http://localhost:8000/process-frame';
  private readonly http = inject(HttpClient)

  processFrame(selectedObject: string, imageFile: File): Observable<any> {
    const formData = new FormData();
    formData.append('selected_object', selectedObject);
    formData.append('file', imageFile);
    return this.http.post<any>(this.apiUrl, formData);
  }
}
