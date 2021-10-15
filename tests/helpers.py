from pandas.testing import assert_frame_equal

def assert_frame_equal_no_index(df1, df2):
    assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True))