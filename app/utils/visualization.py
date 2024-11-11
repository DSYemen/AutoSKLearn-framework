import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64


def create_feature_importance_plot(feature_importance):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importance.values, y=feature_importance.index)
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Features')

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    return f"data:image/png;base64,{plot_data}"


def create_learning_curve_plot(train_sizes, train_scores, test_scores):
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores.mean(axis=1), label='Training score')
    plt.plot(train_sizes,
             test_scores.mean(axis=1),
             label='Cross-validation score')
    plt.title('Learning Curve')
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.legend(loc="best")

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    return f"data:image/png;base64,{plot_data}"

def create_plot(plot_type, *args, **kwargs):
    if plot_type == 'feature_importance':
        plt.figure(figsize=(10, 6))
        sns.barplot(x=kwargs['feature_importance'].values, y=kwargs['feature_importance'].index)
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Features')
    elif plot_type == 'learning_curve':
        plt.figure(figsize=(10, 6))
        plt.plot(kwargs['train_sizes'], kwargs['train_scores'].mean(axis=1), label='Training score')
        plt.plot(kwargs['train_sizes'],
                 kwargs['test_scores'].mean(axis=1),
                 label='Cross-validation score')
        plt.title('Learning Curve')
        plt.xlabel('Training examples')
        plt.ylabel('Score')
        plt.legend(loc="best")
    else:
        raise ValueError("Invalid plot type. Supported types are 'feature_importance' and 'learning_curve'.")

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    return f"data:image/png;base64,{plot_data}"

def create_plot(feature_importance, train_sizes, train_scores, test_scores):
    # دالة لإنشاء رسم بياني لاهمية الميزات
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importance.values, y=feature_importance.index)
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Features')

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    feature_importance_plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    # دالة لإنشاء رسم بياني لمنحنى التعلم
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores.mean(axis=1), label='Training score')
    plt.plot(train_sizes,
             test_scores.mean(axis=1),
             label='Cross-validation score')
    plt.title('Learning Curve')
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.legend(loc="best")

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    learning_curve_plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    return {
        "feature_importance_plot": f"data:image/png;base64,{feature_importance_plot_data}",
        "learning_curve_plot": f"data:image/png;base64,{learning_curve_plot_data}"
    }
