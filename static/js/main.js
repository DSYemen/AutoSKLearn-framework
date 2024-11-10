// static/js/main.js
const app = {
    data() {
        return {
            selectedFile: null,
            processing: false,
            results: null,
            error: null,
            models: [],
            selectedModel: null,
            stats: {
                active_models: 0,
                total_predictions: 0,
                predictions_per_hour: 0,
                avg_accuracy: 0,
                accuracy_trend: 0,
                system_health: 100,
                health_status: 'Healthy',
                new_models_24h: 0
            },
            charts: {}
        }
    },

    mounted() {
        this.initializeCharts();
        this.fetchDashboardData();
        this.startAutoRefresh();
    },

    methods: {
        async handleFileSelect(event) {
            this.selectedFile = event.target.files[0];
            this.error = null;
        },

        async handleFileDrop(event) {
            this.selectedFile = event.dataTransfer.files[0];
            this.error = null;
        },

        formatFileSize(bytes) {
            const units = ['B', 'KB', 'MB', 'GB'];
            let size = bytes;
            let unitIndex = 0;

            while (size >= 1024 && unitIndex < units.length - 1) {
                size /= 1024;
                unitIndex++;
            }

            return `${size.toFixed(1)} ${units[unitIndex]}`;
        },

        async processFile() {
            if (!this.selectedFile) return;

            this.processing = true;
            this.error = null;

            try {
                const formData = new FormData();
                formData.append('file', this.selectedFile);

                const response = await fetch('/api/v1/train', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error(await response.text());
                }

                this.results = await response.json();
                this.updateCharts();

            } catch (error) {
                this.error = error.message;
            } finally {
                this.processing = false;
            }
        },

        async fetchDashboardData() {
            try {
                const [modelsResponse, statsResponse] = await Promise.all([
                    fetch('/api/v1/models'),
                    fetch('/api/v1/stats')
                ]);

                this.models = await modelsResponse.json();
                this.stats = await statsResponse.json();

                this.updateDashboardCharts();
            } catch (error) {
                console.error('Error fetching dashboard data:', error);
            }
        },

        initializeCharts() {
            // Performance Trend Chart
            this.charts.performanceTrend = new Chart(
                document.getElementById('performance-trend-plot'),
                {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Accuracy',
                            data: [],
                            borderColor: '#6366f1',
                            tension: 0.4
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false
                    }
                }
            );

            // Prediction Distribution Chart
            this.charts.predictionDist = new Chart(
                document.getElementById('prediction-distribution-plot'),
                {
                    type: 'bar',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Predictions',
                            data: [],
                            backgroundColor: '#6366f1'
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false
                    }
                }
            );
        },

        updateDashboardCharts() {
            // Update performance trend
            this.charts.performanceTrend.data.labels = 
                this.stats.performance_history.map(p => p.date);
            this.charts.performanceTrend.data.datasets[0].data = 
                this.stats.performance_history.map(p => p.accuracy);
            this.charts.performanceTrend.update();

            // Update prediction distribution
            this.charts.predictionDist.data.labels = 
                Object.keys(this.stats.prediction_distribution);
            this.charts.predictionDist.data.datasets[0].data = 
                Object.values(this.stats.prediction_distribution);
            this.charts.predictionDist.update();
        },

        startAutoRefresh() {
            setInterval(() => {
                this.fetchDashboardData();
            }, 30000); // Refresh every 30 seconds
        },

        async viewModelDetails(modelId) {
            try {
                const response = await fetch(`/api/v1/models/${modelId}`);
                this.selectedModel = await response.json();
            } catch (error) {
                console.error('Error fetching model details:', error);
            }
        },

        formatDate(date) {
            return new Date(date).toLocaleString();
        },

        formatMetricName(metric) {
            return metric.split('_')
                .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                .join(' ');
        },

        formatMetricValue(value) {
            if (typeof value === 'number') {
                return value.toFixed(4);
            }
            return value;
        },

        async downloadModel(modelId) {
            window.location.href = `/api/v1/models/${modelId}/download`;
        },

        async retrainModel(modelId) {
            try {
                await fetch(`/api/v1/models/${modelId}/retrain`, {
                    method: 'POST'
                });
                this.fetchDashboardData();
            } catch (error) {
                console.error('Error retraining model:', error);
            }
        }
    }
};

Vue.createApp(app).mount('#app');