import { Component, OnInit } from '@angular/core';
import { ChildrenService, Child } from '../children.service';

@Component({
  selector: 'app-thirty',
  templateUrl: './thirty.component.html',
  styleUrls: ['./thirty.component.css']
})
export class ThirtyComponent implements OnInit {
  children: Child[] = [];

  constructor(private childrenService: ChildrenService) {}

  ngOnInit(): void {
    this.children = this.childrenService.getChildrenBornIn30DayMonths();
  }
}
