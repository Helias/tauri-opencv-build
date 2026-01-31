import { ChangeDetectionStrategy, Component } from '@angular/core';

@Component({
  selector: 'app-guide',
  templateUrl: './guide.component.html',
  styleUrls: ['./guide.component.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class GuideComponent {}
