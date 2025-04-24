"""
Data preprocessing functionality for the ML Platform.
"""

from preprocessing.missing_values import (
    handle_missing_values,
    remove_missing_rows,
    remove_missing_columns,
    fill_missing_mean,
    fill_missing_median,
    fill_missing_mode,
    fill_missing_constant,
    fill_missing_interpolation,
    fill_missing_knn
)

from preprocessing.outliers import (
    detect_outliers,
    remove_outliers,
    cap_outliers,
    replace_outliers_mean,
    replace_outliers_median
)

from preprocessing.encoding import (
    encode_features,
    one_hot_encode,
    label_encode,
    ordinal_encode,
    target_encode,
    binary_encode,
    frequency_encode
)

from preprocessing.scaling import (
    scale_features,
    standardize,
    minmax_scale,
    robust_scale,
    normalize
)

from preprocessing.image_preprocessing import (
    resize_image,
    normalize_image,
    augment_image
)

__all__ = [
    # Missing values
    'handle_missing_values',
    'remove_missing_rows',
    'remove_missing_columns',
    'fill_missing_mean',
    'fill_missing_median',
    'fill_missing_mode',
    'fill_missing_constant',
    'fill_missing_interpolation',
    'fill_missing_knn',
    
    # Outliers
    'detect_outliers',
    'remove_outliers',
    'cap_outliers',
    'replace_outliers_mean',
    'replace_outliers_median',
    
    # Encoding
    'encode_features',
    'one_hot_encode',
    'label_encode',
    'ordinal_encode',
    'target_encode',
    'binary_encode',
    'frequency_encode',
    
    # Scaling
    'scale_features',
    'standardize',
    'minmax_scale',
    'robust_scale',
    'normalize',
    
    # Image preprocessing
    'resize_image',
    'normalize_image',
    'augment_image'
]
