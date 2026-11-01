# Usage: make Assignment_2   (or `make` for every Assignment_*/ with a report.md)
# Output: Assignment_N/dist/report.pdf
#         Assignment_N/dist/Assignment_N.zip (report.pdf + src/)

ASSIGNMENTS := $(patsubst %/report.md,%,$(wildcard Assignment_*/report.md))

.PHONY: all clean $(ASSIGNMENTS)

all: $(ASSIGNMENTS)

$(ASSIGNMENTS):
	@mkdir -p $@/dist
	@rm -f $@/dist/$@.zip
	cd $@ && npx --yes md-to-pdf report.md
	mv $@/report.pdf $@/dist/report.pdf
	cd $@ && find src -type d -name build -prune -o -type f \
		\( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name CMakeLists.txt \) -print \
		| zip dist/$@.zip -@
	cd $@/dist && zip $@.zip report.pdf

clean:
	rm -rf $(addsuffix /dist,$(ASSIGNMENTS))
