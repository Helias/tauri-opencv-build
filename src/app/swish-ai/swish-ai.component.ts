import { DecimalPipe } from '@angular/common';
import {
  Component,
  DestroyRef,
  inject,
  OnDestroy,
  OnInit,
} from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { FormsModule } from '@angular/forms';
import { save } from '@tauri-apps/plugin-dialog';
import { writeFile } from '@tauri-apps/plugin-fs';
import { fetch } from '@tauri-apps/plugin-http';
import { interval, Subscription } from 'rxjs';
import { TauriService } from '../core/services';
import { SwishAiService } from '../core/services/swish-ai/swish-ai.service';
import { GuideComponent, HeaderComponent } from '../shared/components';
import { FooterComponent } from '../shared/components/footer/footer.component';
import {
  DEFAULT_THRESHOLDS,
  LABELS,
  ProcessingMode,
  Stats,
  StatusResponse,
  Thresholds,
  UploadResponse,
} from '../shared/models/swish-ai.models';

type Status =
  | 'idle'
  | 'uploading'
  | 'processing'
  | 'completed'
  | 'error'
  | 'stopped';

@Component({
  selector: 'app-swish-ai',
  imports: [
    DecimalPipe,
    FormsModule,
    GuideComponent,
    HeaderComponent,
    FooterComponent,
  ],
  templateUrl: './swish-ai.component.html',
  styleUrls: ['./swish-ai.component.scss'],
})
export class SwishAiComponent implements OnInit, OnDestroy {
  protected readonly tauriService = inject(TauriService);
  private readonly destroyRef = inject(DestroyRef);

  // File state
  file: File | null = null;
  fileId: string | null = null;

  // Status state
  status: Status = 'idle';
  progress: number = 0;
  stats: Stats = { shots: 0, baskets: 0, accuracy: 0 };

  // Settings
  testMode: boolean = false;
  processingMode: ProcessingMode = 'full_tracking';

  // Advanced settings
  showAdvancedSettings: boolean = false;
  thresholds: Thresholds = { ...DEFAULT_THRESHOLDS };

  // Error handling
  errorMsg: string = '';

  // Status polling
  private statusSubscription: Subscription | null = null;
  private progressSubscription: Subscription | null = null;
  private selectedFilePath: string | null = null;
  private outputFilePath: string | null = null;

  // Constants for template
  readonly LABELS = LABELS;
  readonly DEFAULT_THRESHOLDS = DEFAULT_THRESHOLDS;

  private swishAiService = inject(SwishAiService);

  ngOnInit(): void {
    if (this.tauriService.isTauri) {
      this.setupTauriProgressListener();
    }
  }

  ngOnDestroy(): void {
    this.stopStatusPolling();
    if (this.progressSubscription) {
      this.progressSubscription.unsubscribe();
    }
  }

  private setupTauriProgressListener(): void {
    // Keep event-based progress for backward compatibility
    this.progressSubscription = this.tauriService.progress$
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((progressData) => {
        this.progress = progressData.progress;

        if (progressData.shots_attempted !== undefined) {
          this.stats = {
            shots: progressData.shots_attempted,
            baskets: progressData.baskets_made || 0,
            accuracy: progressData.accuracy || 0,
          };
        }

        if (progressData.progress >= 100 && progressData.output_path) {
          this.status = 'completed';
          this.outputFilePath = progressData.output_path;
        }
      });
  }

  private startTauriProgressPolling(): void {
    // Poll progress every second
    this.statusSubscription = interval(1000)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe(() => {
        this.checkTauriProgress();
      });
  }

  private checkTauriProgress(): void {
    this.swishAiService
      .getProcessingProgress()
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (progressData) => {
          this.progress = progressData.progress;

          if (progressData.shots_attempted !== undefined) {
            this.stats = {
              shots: progressData.shots_attempted,
              baskets: progressData.baskets_made || 0,
              accuracy: progressData.accuracy || 0,
            };
          }

          if (progressData.progress >= 100 && progressData.output_path) {
            this.status = 'completed';
            this.outputFilePath = progressData.output_path;
            this.stopStatusPolling();
          }
        },
        error: (err: any) => {
          console.error('Progress check failed', err);
        },
      });
  }

  protected onClickSelectFile(event: Event): void {
    if (this.tauriService.isTauri) {
      this.onFileChange(event);
      event.preventDefault();
    }
  }
  async onFileChange(event: Event): Promise<void> {
    if (this.tauriService.isTauri) {
      const filePath = await this.swishAiService.selectVideoFile();
      if (filePath) {
        console.log('Selected file path:', filePath);
        this.selectedFilePath = filePath;
        this.file = { name: filePath.split('/').pop() || filePath } as File;
        this.status = 'idle';
      }
      return;
    }

    const input = event.target as HTMLInputElement;
    if (input.files && input.files[0]) {
      this.file = input.files[0];
      this.status = 'idle';
      this.progress = 0;
      this.stats = { shots: 0, baskets: 0, accuracy: 0 };
      this.errorMsg = '';
    }
  }

  resetApp(): void {
    this.file = null;
    this.fileId = null;
    this.status = 'idle';
    this.progress = 0;
    this.stats = { shots: 0, baskets: 0, accuracy: 0 };
    this.errorMsg = '';
    this.testMode = false;
    this.stopStatusPolling();
  }

  resetThresholds(): void {
    this.thresholds = { ...DEFAULT_THRESHOLDS };
  }

  updateThreshold(key: string, value: string): void {
    this.thresholds = {
      ...this.thresholds,
      [key]: parseFloat(value),
    };
  }

  getThresholdKeys(): string[] {
    return Object.keys(DEFAULT_THRESHOLDS);
  }

  uploadAndStart(): void {
    if (!this.file) return;

    // Tauri mode
    if (this.tauriService.isTauri && this.selectedFilePath) {
      this.status = 'processing';
      this.errorMsg = '';
      this.progress = 0;

      this.swishAiService
        .processVideoInTauri(this.selectedFilePath)
        .then(() => {
          console.log('Processing started...');
          // Start polling for progress
          this.startTauriProgressPolling();
        })
        .catch((err) => {
          console.error(err);
          this.status = 'error';
          this.errorMsg = err.message || 'Processing failed';
        });

      return;
    }

    // Web mode (existing code)
    this.status = 'uploading';
    this.errorMsg = '';

    this.swishAiService
      .uploadVideo(this.file)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (uploadData: UploadResponse) => {
          this.fileId = uploadData.file_id;

          if (!this.fileId) {
            this.status = 'error';
            this.errorMsg = 'Invalid file ID received';
            return;
          }

          this.swishAiService
            .startProcessing(
              this.fileId,
              this.testMode,
              this.processingMode,
              this.thresholds,
            )
            .pipe(takeUntilDestroyed(this.destroyRef))
            .subscribe({
              next: () => {
                this.status = 'processing';
                this.startStatusPolling();
              },
              error: (err: any) => {
                console.error(err);
                this.status = 'error';
                this.errorMsg = 'Processing start failed';
              },
            });
        },
        error: (err: any) => {
          console.error(err);
          this.status = 'error';
          this.errorMsg = 'Upload failed. Check connection or file size.';
        },
      });
  }

  private startStatusPolling(): void {
    this.statusSubscription = interval(1000)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe(() => {
        this.checkStatus();
      });
  }

  private stopStatusPolling(): void {
    if (this.statusSubscription) {
      this.statusSubscription.unsubscribe();
      this.statusSubscription = null;
    }
  }

  private checkStatus(): void {
    if (!this.fileId) return;

    this.swishAiService
      .getStatus(this.fileId)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (data: StatusResponse) => {
          if (data.status === 'processing') {
            this.progress = data.percentage;
            this.stats = data.stats;
          } else if (data.status === 'completed') {
            this.status = 'completed';
            this.progress = 100;
            this.stats = data.stats;
            this.stopStatusPolling();
          } else if (data.status === 'error') {
            this.status = 'error';
            this.errorMsg = data.message || 'An unknown error occurred';
            this.stopStatusPolling();
          } else if (data.status === 'stopped') {
            this.status = 'stopped';
            this.stopStatusPolling();
          }
        },
        error: (err: any) => {
          console.error('Status check failed', err);
        },
      });
  }

  stopProcessing(): void {
    if (!this.fileId) return;

    this.swishAiService
      .stopProcessing(this.fileId)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: () => {
          this.status = 'stopped';
          this.stopStatusPolling();
        },
        error: (err: any) => {
          console.error(err);
        },
      });
  }

  async downloadVideo(): Promise<void> {
    // In Tauri mode, prompt user to save the file from temp location
    if (this.tauriService.isTauri && this.outputFilePath) {
      try {
        // Ask user where to save the file
        const savePath = await save({
          defaultPath: 'swishai_output.mp4',
          filters: [
            {
              name: 'Video',
              extensions: ['mp4'],
            },
          ],
        });

        if (!savePath) {
          // User cancelled
          return;
        }

        // Copy the file from temp location to user-selected location
        await this.tauriService.copyFile(this.outputFilePath, savePath);

        console.log('Video saved to:', savePath);
        alert(`Video successfully saved to:\n${savePath}`);
      } catch (error) {
        console.error('Save error:', error);
        this.errorMsg = 'Failed to save video';
      }
      return;
    }

    // Web mode - download from API
    if (!this.fileId) return;

    try {
      // Ask user where to save the file
      const filePath = await save({
        defaultPath: `processed_video_${this.fileId}.mp4`,
        filters: [
          {
            name: 'Video',
            extensions: ['mp4'],
          },
        ],
      });

      if (!filePath) return; // User cancelled

      // Download the video from the API
      const response = await fetch(
        this.swishAiService.getDownloadUrl(this.fileId),
        {
          method: 'GET',
        },
      );

      if (!response.ok) {
        throw new Error('Download failed');
      }

      // Get the video data as array buffer
      const videoData = await response.arrayBuffer();

      // Write the file to disk
      await writeFile(filePath, new Uint8Array(videoData));

      console.log('Video downloaded successfully to:', filePath);
    } catch (error) {
      console.error('Download error:', error);
      this.errorMsg = 'Failed to download video';
    }
  }

  toggleAdvancedSettings(): void {
    this.showAdvancedSettings = !this.showAdvancedSettings;
  }
}
