import { Injectable } from '@angular/core';

export interface Child {
  name: string;
  monthBorn: string;
  age: number;
}

@Injectable({
  providedIn: 'root'
})
export class ChildrenService {
  private children: Child[] = [
    { name: 'Alice', monthBorn: 'January', age: 7 },
    { name: 'Bob', monthBorn: 'February', age: 5 },
    { name: 'Charlie', monthBorn: 'March', age: 8 },
    { name: 'Diana', monthBorn: 'April', age: 6 },
    { name: 'Eve', monthBorn: 'May', age: 9 },
    { name: 'Frank', monthBorn: 'June', age: 4 },
    { name: 'Grace', monthBorn: 'July', age: 7 },
    { name: 'Henry', monthBorn: 'August', age: 10 },
    { name: 'Ivy', monthBorn: 'September', age: 6 },
    { name: 'Jack', monthBorn: 'October', age: 8 },
    { name: 'Kate', monthBorn: 'November', age: 5 },
    { name: 'Leo', monthBorn: 'December', age: 7 }
  ];

  // Months with 30 days (including February as per the question)
  private thirtyDayMonths: string[] = [
    'February', 'April', 'June', 'September', 'November'
  ];

  // Months with 31 days
  private thirtyOneDayMonths: string[] = [
    'January', 'March', 'May', 'July', 'August', 'October', 'December'
  ];

  getAllChildren(): Child[] {
    return this.children;
  }

  getChildrenBornIn30DayMonths(): Child[] {
    return this.children.filter(child =>
      this.thirtyDayMonths.includes(child.monthBorn)
    );
  }

  getChildrenBornIn31DayMonths(): Child[] {
    return this.children.filter(child =>
      this.thirtyOneDayMonths.includes(child.monthBorn)
    );
  }
}
