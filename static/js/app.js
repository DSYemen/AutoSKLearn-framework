// static/js/app.js departected  مهمل هذا الملف
const app = {
    data() {
        return {
            currentStep: 1,
            loading: false,
            file: null,
            results: null,
            modelInfo: null,
            predictionInput: {},
            predictions: [],
            error: null
        }
    },
    methods: {
        async processFile() {
            try {
                this.loading = true
                const formData = new FormData()
                formData.append('file', this.file)

                const response = await fetch('/api/process', {
                    method: 'POST',
                    body: formData
                })

                if (!response.ok) throw new Error('Processing failed')

                this.results = await response.json()
                this.currentStep = 2
            } catch (error) {
                this.error = error.message
            } finally {
                this.loading = false
            }
        },

        async makePrediction() {
            // Implementation
        },

        downloadModel() {
            // Implementation
        }
    }
}