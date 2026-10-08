import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'Clock Angle Calculator';
  hour: number | null = null;
  minute: number | null = null;
  angle: number | null = null;
  errorMessage: string = '';

  calculateAngle(): void {
    this.errorMessage = '';
    this.angle = null;

    // Form validation
    if (this.hour === null || this.minute === null) {
      this.errorMessage = 'Please enter both hour and minute values.';
      return;
    }
    if (this.hour < 1 || this.hour > 12) {
      this.errorMessage = 'Hour must be between 1 and 12.';
      return;
    }
    if (this.minute < 0 || this.minute > 59) {
      this.errorMessage = 'Minute must be between 0 and 59.';
      return;
    }

    // Calculate the angle of the hour hand from 12 o'clock position
    // Hour hand moves 0.5 degrees per minute (360/12/60)
    const hourAngle = (this.hour % 12) * 30 + this.minute * 0.5;

    // Calculate the angle of the minute hand from 12 o'clock position
    // Minute hand moves 6 degrees per minute (360/60)
    const minuteAngle = this.minute * 6;

    // Find the absolute difference
    let diff = Math.abs(hourAngle - minuteAngle);

    // Take the smaller angle (clock has two arcs)
    if (diff > 180) {
      diff = 360 - diff;
    }

    this.angle = diff;
  }

  reset(): void {
    this.hour = null;
    this.minute = null;
    this.angle = null;
    this.errorMessage = '';
  }
}
