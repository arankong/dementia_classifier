from dementia_classifier.feature_extraction import save_dbank_to_sql, save_blog_to_sql
from dementia_classifier.analysis import feature_set, blog


def save_features_to_database():
    save_dbank_to_sql.save_all_to_sql()
    save_blog_to_sql.save_blog_data()


def save_all_results():
    print "-----------------------------------------"
    print "Saving: save_new_feature_results_to_sql()"
    print "-----------------------------------------"


    print "-----------------------------------------"
    print "Saving: save_blog_results_to_sql()"
    print "-----------------------------------------"
    blog.save_blog_results_to_sql()


def save_all_plots():
    # New features
    feature_set.new_feature_set_plot(metric='fms', absolute=True,  poly=False)
    feature_set.new_feature_set_plot(metric='fms', absolute=True,  poly=True)
    feature_set.new_feature_set_plot(metric='fms', absolute=False, poly=True)
    feature_set.new_feature_set_plot(metric='fms', absolute=False, poly=False)

    # New feature appendix
    feature_set.new_feature_set_plot(metric='fms', absolute=False, poly=False)
    feature_set.new_feature_set_plot(metric='acc', absolute=True)
    feature_set.new_feature_set_plot(metric='acc', absolute=False, poly=True)
    feature_set.new_feature_set_plot(metric='roc', absolute=True)
    feature_set.new_feature_set_plot(metric='roc', absolute=False, poly=True)

    # Blog
    blog.blog_plot()


def main():
    save_features_to_database()
    save_all_results()
    save_all_plots()


if __name__ == '__main__':
    main()
