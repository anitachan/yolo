import { AfterViewInit, Component, ElementRef, inject, Inject, OnDestroy, OnInit, PLATFORM_ID, ViewChild } from '@angular/core';
import { ImageProcessingService } from '../../services/image-processing.service';
import { FormsModule } from '@angular/forms';
import { CommonModule, isPlatformBrowser } from '@angular/common';
import { Subscription } from 'rxjs';

@Component({
  selector: 'app-webcam',
  imports: [CommonModule, FormsModule],
  standalone: true,
  templateUrl: './webcam.component.html',
  styleUrl: './webcam.component.scss'
})
export class WebcamComponent implements OnInit, OnDestroy, AfterViewInit {
  @ViewChild('video', { static: false }) video!: ElementRef;
  @ViewChild('canvas', { static: true }) canvas!: ElementRef;

  subscription = new Subscription();

  fixedObjects = ["person", "cell phone", "bicycle", "car", "dog", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "bottle", "cup", "spoon", "chair", "door", "mouse", "keyboard", "laptop", "tv", "book"];
  selectedObject = this.fixedObjects[0];

  availableCameras: MediaDeviceInfo[] = [];
  selectedCamera: string = '';

  processedImage: string = '';
  processing: boolean = false;
  processingInterval: any;
  videoStream: MediaStream | null = null;

  private readonly imageProcessingService = inject(ImageProcessingService);
  private readonly platformId = inject(PLATFORM_ID);

  constructor(
  ) { }

  ngOnInit() {
    if (isPlatformBrowser(this.platformId)) {
      this.getCameras();
    }
  }

  ngAfterViewInit() {
    if (!this.processing && this.selectedCamera) {
      this.startCameraStream();
    }
  }

  ngOnDestroy() {
    this.subscription?.unsubscribe();
    if (this.processingInterval) {
      clearInterval(this.processingInterval);
    }
    if (this.videoStream) {
      this.videoStream.getTracks().forEach(track => track.stop());
    }
  }

  getCameras() {
    navigator.mediaDevices.enumerateDevices()
      .then(devices => {
        this.availableCameras = devices.filter(device => device.kind === 'videoinput');
        if (this.availableCameras.length > 0) {
          this.selectedCamera = this.availableCameras[0].deviceId;
          this.startCameraStream();
        }
      })
      .catch(err => console.error("Error enumerating devices:", err));
  }

  startCameraStream() {
    if (isPlatformBrowser(this.platformId) && this.selectedCamera) {
      const constraints = {
        video: { deviceId: { exact: this.selectedCamera } }
      };
      navigator.mediaDevices.getUserMedia(constraints)
        .then(stream => {
          this.videoStream = stream;
          if (this.video?.nativeElement) {
            this.video.nativeElement.srcObject = stream;
          }
        })
        .catch(err => console.error("Error accessing webcam:", err));
    }
  }

  toggleProcessing() {
    this.processing = !this.processing;
    if (this.processing) {
      this.processingInterval = setInterval(() => {
        this.captureAndProcessFrame();
      }, 1000);
    } else {
      clearInterval(this.processingInterval);
      if (this.videoStream && this.video?.nativeElement) {
        this.video.nativeElement.srcObject = this.videoStream;
      }
    }
  }

  captureAndProcessFrame() {
    // Use the video element to capture the current frame
    if (!this.video?.nativeElement) {
      return;
    }
    const context = this.canvas.nativeElement.getContext('2d');
    context.drawImage(this.video.nativeElement, 0, 0, 640, 480);
    this.canvas.nativeElement.toBlob((blob: any) => {
      if (blob) {
        const imageFile = new File([blob], 'frame.jpg', { type: 'image/jpeg' });
        this.subscription.add(this.imageProcessingService.processFrame(this.selectedObject, imageFile)
          .subscribe(response => {
            this.processedImage = response.processed_image;
            console.log("Detections:", response.detections);
          }));
      }
    }, 'image/jpeg');
  }
}
