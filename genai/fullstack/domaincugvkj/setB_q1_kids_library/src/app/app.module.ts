import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { AppComponent } from './app.component';
import { ThirtyComponent } from './thirty/thirty.component';
import { ThirtyOneComponent } from './thirtyone/thirtyone.component';
import { ChildrenService } from './children.service';

@NgModule({
  declarations: [
    AppComponent,
    ThirtyComponent,
    ThirtyOneComponent
  ],
  imports: [BrowserModule],
  providers: [ChildrenService],
  bootstrap: [AppComponent]
})
export class AppModule {}
