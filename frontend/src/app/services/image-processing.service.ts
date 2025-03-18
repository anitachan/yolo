import { Injectable } from '@angular/core';
import { Subject } from 'rxjs';


export interface ProcessedData {
  processed_image: string;
  detections: any[];
}

@Injectable({
  providedIn: 'root'
})
export class ImageProcessingService {
  private ws!: WebSocket;
  public processedData$ = new Subject<ProcessedData>();

  connect(): void {
    this.ws = new WebSocket("ws://localhost:8000/ws");
    this.ws.onmessage = (message) => {
      try {
        const data: ProcessedData = JSON.parse(message.data);
        this.processedData$.next(data);
      } catch (e) {
        console.error("Error parsing WebSocket message", e);
      }
    };
    this.ws.onerror = (error) => {
      console.error("WebSocket error:", error);
    };
  }

  sendFrame(frameBlob: Blob): void {
    frameBlob.arrayBuffer().then(buffer => {
      const uint8Buffer = new Uint8Array(buffer);
      this.ws.send(uint8Buffer);
    });
  }

  sendSelectedObject(selectedObject: string): void {
    const msg = JSON.stringify({ selected_object: selectedObject });
    this.ws.send(msg);
  }
}
