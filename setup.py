from setuptools import find_packages, setup

package_name = "social_companion"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="emon",
    maintainer_email="moemon_miea.hgwgst@n.w.rd.honda.co.jp",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "social_companion_wrapper = social_companion.social_companion_wrapper:main",
            "generate_dummy_pedestrians = social_companion.pub:main",
        ],
    },
)
