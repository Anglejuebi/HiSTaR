from setuptools import setup

def load_requirements():
    try:
        with open("requirements.txt", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        print("error! Cannot find requirements.txt file")
        return []

if __name__ == "__main__":
    setup(
        name="HiSTaR",
        version="1.0.0",
        description="HiSTaR: identifying spatial domains with Hierarchical Spatial Transcriptomics variational autoencoder",
        url="https://github.com/Anglejuebi/HDVAE",
        author="JunHua Yu",
        author_email="Angle_Yu@e.gzhu.edu.cn",
        license="MIT",
        packages=["HiSTaR"],
        install_requires=load_requirements(),
        zip_safe=False,
        include_package_data=True,
        long_description=""" in this paper, we propose a Hierarchical ST variational autoencoder (HiSTaR) to extract multi-level latent features of spots. HiSTaR tends to perform well in identifying spatial domains across multiple datasets from diverse platforms, consistently showing superior results compared to existing methods.""",
        long_description_content_type="text/markdown",
    )
