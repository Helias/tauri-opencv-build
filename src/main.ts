import {
  provideHttpClient,
  withInterceptorsFromDi,
} from '@angular/common/http';
import {
  enableProdMode,
  importProvidersFrom,
  provideZoneChangeDetection,
} from '@angular/core';
import { bootstrapApplication } from '@angular/platform-browser';
import { provideRouter } from '@angular/router';

import { provideTranslateService } from '@ngx-translate/core';
import { provideTranslateHttpLoader } from '@ngx-translate/http-loader';
import { AppComponent } from './app/app.component';
import { CoreModule } from './app/core/core.module';
import { PageNotFoundComponent } from './app/shared/components';
import { SwishAiComponent } from './app/swish-ai/swish-ai.component';
import { APP_CONFIG } from './environments/environment';

if (APP_CONFIG.production) {
  enableProdMode();
}

bootstrapApplication(AppComponent, {
  providers: [
    provideZoneChangeDetection(),
    provideHttpClient(withInterceptorsFromDi()),
    provideTranslateService({
      loader: provideTranslateHttpLoader({
        prefix: './assets/i18n/',
        suffix: '.json',
      }),
      fallbackLang: 'en',
      lang: 'en',
    }),
    provideRouter([
      {
        path: '',
        redirectTo: 'swish-ai',
        pathMatch: 'full',
      },
      {
        path: 'swish-ai',
        component: SwishAiComponent,
      },
      {
        path: '**',
        component: PageNotFoundComponent,
      },
    ]),
    importProvidersFrom(CoreModule),
  ],
}).catch((err) => console.error(err));
