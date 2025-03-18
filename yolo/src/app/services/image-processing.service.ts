import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ImageProcessingService {

  private apiUrl = 'http://localhost:8000/process-frame';

  constructor(private http: HttpClient) { }

  processFrame(selectedObject: string, imageFile: File): Observable<any> {
    const formData = new FormData();
    formData.append('selected_object', selectedObject);
    formData.append('file', imageFile);
    return this.http.post<any>(this.apiUrl, formData);
  }
}
