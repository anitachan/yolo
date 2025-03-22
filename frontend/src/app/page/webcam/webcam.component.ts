import {
  AfterViewInit,
  Component,
  ElementRef,
  inject,
  OnDestroy,
  OnInit,
  PLATFORM_ID,
  ViewChild
} from '@angular/core';
import { ImageProcessingService, ProcessedData } from '../../services/image-processing.service';
import { FormsModule } from '@angular/forms';
import { CommonModule, isPlatformBrowser } from '@angular/common';
import { Subscription } from 'rxjs';

@Component({
  selector: 'app-webcam',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './webcam.component.html',
  styleUrl: './webcam.component.scss'
})
export class WebcamComponent implements OnInit, OnDestroy, AfterViewInit {
  @ViewChild('video', { static: false }) video!: ElementRef<HTMLVideoElement>;
  @ViewChild('canvas', { static: true }) canvas!: ElementRef<HTMLCanvasElement>;

  fixedObjects = ["person", "cell phone", "bicycle", "car", "dog", "backpack"];
  selectedObject = this.fixedObjects[0];

  availableCameras: MediaDeviceInfo[] = [];
  selectedCamera: string = '';
  processedImage: string = '';
  detections: any[] = [];
  videoStream: MediaStream | null = null;
  loading: boolean = false;
  statusMessage: string = '';
  errorMessage: string = '';
  cameraPermissionGranted: boolean = false;

  isDetecting: boolean = false;

  private readonly imageProcessingService = inject(ImageProcessingService);
  private readonly platformId = inject(PLATFORM_ID);
  private readonly subscription = new Subscription();
  private processingInterval: any;
  private processingInProgress = false;

  ngOnInit(): void {
    if (!isPlatformBrowser(this.platformId)) return;

    this.statusMessage = "Initializing...";
    this.getCameras();
    this.imageProcessingService.connect();

    this.subscription.add(
      this.imageProcessingService.processedData$.subscribe((data: ProcessedData) => {
        this.processingInProgress = false;
        this.processedImage = data.processed_image;
        this.detections = data.detections.map(d => ({
          label: d.label,
          classification: d.classification
        }));
        this.statusMessage = "Frame processed.";
      })
    );

    this.imageProcessingService.sendSelectedObject(this.selectedObject);
  }

  ngAfterViewInit(): void {
    if (!this.processingInterval && this.selectedCamera) {
      this.startCameraStream();
    }
  }

  ngOnDestroy(): void {
    this.subscription.unsubscribe();
    if (this.processingInterval) clearInterval(this.processingInterval);
    if (this.videoStream) this.videoStream.getTracks().forEach(track => track.stop());
    this.imageProcessingService.disconnect();
  }

  requestCameraPermission(): void {
    if (!isPlatformBrowser(this.platformId)) return;

    this.statusMessage = "Requesting camera access...";
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        this.cameraPermissionGranted = true;
        this.startCameraStream();
        this.statusMessage = "Camera access granted.";
      })
      .catch(() => {
        this.errorMessage = "Camera access denied.";
        this.statusMessage = "";
      });
  }

  getCameras(): void {
    if (!isPlatformBrowser(this.platformId)) return;

    navigator.mediaDevices.enumerateDevices()
      .then(devices => {
        this.availableCameras = devices.filter(device => device.kind === 'videoinput');
        if (this.availableCameras.length > 0) {
          this.selectedCamera = this.availableCameras[0].deviceId;
          this.requestCameraPermission();
        } else {
          this.errorMessage = "No cameras found.";
        }
      })
      .catch(() => {
        this.errorMessage = "Error accessing camera devices.";
      });
  }

  startCameraStream(): void {
    if (!isPlatformBrowser(this.platformId) || !this.selectedCamera) return;

    this.loading = true;
    this.statusMessage = "Starting webcam...";
    const constraints = { video: { deviceId: { exact: this.selectedCamera } } };

    navigator.mediaDevices.getUserMedia(constraints)
      .then(stream => {
        this.videoStream = stream;
        if (this.video?.nativeElement) {
          this.video.nativeElement.srcObject = stream;
        }
        this.loading = false;
        this.statusMessage = "Webcam running.";
      })
      .catch(() => {
        this.errorMessage = "Error accessing webcam.";
        this.statusMessage = "";
        this.loading = false;
      });
  }

  startDetection(): void {
    if (!this.processingInterval) {
      this.statusMessage = "Detección iniciada.";
      this.processingInterval = setInterval(() => this.captureAndSendFrame(), 33);
      this.isDetecting = true;
    }
  }

  pauseDetection(): void {
    if (this.processingInterval) {
      clearInterval(this.processingInterval);
      this.processingInterval = null;
      this.statusMessage = "Detección pausada.";
      this.isDetecting = false;
    }
  }

  captureAndSendFrame(): void {
    if (!this.video?.nativeElement || this.processingInProgress) return;

    const context = this.canvas.nativeElement.getContext('2d');
    context?.drawImage(this.video.nativeElement, 0, 0, 640, 480);

    this.canvas.nativeElement.toBlob((blob: Blob | null) => {
      if (blob) {
        this.processingInProgress = true;
        this.imageProcessingService.sendFrame(blob);
      }
    }, 'image/jpeg');
  }

  takeScreenshot(): void {
    if (!this.processedImage) return;

    const a = document.createElement('a');
    a.href = 'data:image/jpeg;base64,' + this.processedImage;
    a.download = `captura-${Date.now()}.jpg`;
    a.click();
  }

  onSelectedObjectChange(): void {
    this.imageProcessingService.sendSelectedObject(this.selectedObject);
    this.statusMessage = `Object changed to \"${this.selectedObject}\"`;
  }
}
