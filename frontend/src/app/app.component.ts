import { Component } from '@angular/core';
import { WebcamComponent } from './page/webcam/webcam.component';

@Component({
  selector: 'app-root',
  imports: [WebcamComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss'
})
export class AppComponent {
  title = 'yolo';
}
