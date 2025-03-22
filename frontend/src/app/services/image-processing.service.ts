import { Injectable } from '@angular/core';
import { Subject } from 'rxjs';

export interface ProcessedData {
  processed_image: string;
  detections: any[];
}

@Injectable({ providedIn: 'root' })
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

    this.ws.onclose = () => console.warn("WebSocket closed.");
    this.ws.onerror = (error) => console.error("WebSocket error:", error);
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
    }
  }

  sendFrame(frameBlob: Blob): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    frameBlob.arrayBuffer().then(buffer => {
      this.ws.send(new Uint8Array(buffer));
    });
  }

  sendSelectedObject(selectedObject: string): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    this.ws.send(JSON.stringify({ selected_object: selectedObject }));
  }
}
