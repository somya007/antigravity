import { Component, OnInit } from '@angular/core';
import { ChildrenService, Child } from '../children.service';

@Component({
  selector: 'app-thirtyone',
  templateUrl: './thirtyone.component.html',
  styleUrls: ['./thirtyone.component.css']
})
export class ThirtyOneComponent implements OnInit {
  children: Child[] = [];

  constructor(private childrenService: ChildrenService) {}

  ngOnInit(): void {
    this.children = this.childrenService.getChildrenBornIn31DayMonths();
  }
}
