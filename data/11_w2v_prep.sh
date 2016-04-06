#!/usr/bin/env bash
#
# pre-process data to be fed to word2vec
# * EmpiriST 2016 data
# * de.wikipedia.org data
#

set -e

# adapted from word2vec/demo-train-big-model-v1.sh
process_data() {
    awk '{if ($0 == "<s>") {} else if ($0 == "</s>") { print "" } else { printf "%s ",tolower($0)}}' | \
    perl -e '
    # Program to filter Wikipedia XML dumps to "clean" text consisting only of lowercase
    # letters (a-z, converted from A-Z), and spaces (never consecutive)...
    # All other characters are converted to spaces.  Only text which normally appears.
    # in the web browser is displayed.  Tables are removed.  Image captions are.
    # preserved.  Links are converted to normal text.  Digits are spelled out.
    # *** Modified to not spell digits or throw away non-ASCII characters ***

    # Written by Matt Mahoney, June 10, 2006.  This program is released to the public domain.

    while (<>) {
        # Remove any text not normally visible
        s/<.*>//;               # remove xml tags
        s/&amp;/&/g;            # decode URL encoded chars
        s/&lt;/</g;
        s/&gt;/>/g;
        s/<ref[^<]*<\/ref>//g;  # remove references <ref...> ... </ref>
        s/<[^>]*>//g;           # remove xhtml tags
        s/\[http:[^] ]*/[/g;    # remove normal url, preserve visible text
        s/\|thumb//ig;          # remove images links, preserve caption
        s/\|left//ig;
        s/\|right//ig;
        s/\|\d+px//ig;
        s/\[\[image:[^\[\]]*\|//ig;
        s/\[\[category:([^|\]]*)[^]]*\]\]/[[$1]]/ig;  # show categories without markup
        s/\[\[[a-z\-]*:[^\]]*\]\]//g;  # remove links to other languages
        s/\[\[[^\|\]]*\|/[[/g;  # remove wiki url, preserve visible text
        s/{{[^}]*}}//g;         # remove {{icons}} and {tables}
        s/{[^}]*}//g;
        s/\[//g;                # remove [ and ]
        s/\]//g;
        s/&[^;]*;/ /g;          # remove URL encoded chars
        s/[0-9]/0/g;

        $_=" $_ ";
        chop;
        print $_;
    }
    ' | awk '{if ((length($0) > 20) && (NF > 5)) print;}' | sed -e 's/^ //'
}

cat <(cat ./empirist/empirist_training_{cmc,web}/tokenized/*.txt \
      | sed -e 's#^<posting.*#<s>#' -e 's#<article.*#<s>#' -e 's#^$#</s>#') \
    > ./tmp/pre-w2v_empirist.txt
process_data < ./tmp/pre-w2v_empirist.txt > ./tmp/w2v_empirist.txt

cat <(bzip2 -c -d ./tmp/de.wikipedia.org/*.tt.bz2) \
    > ./tmp/pre-w2v_de.wikipedia.org.txt
process_data < ./tmp/pre-w2v_de.wikipedia.org.txt > ./tmp/w2v_de.wikipedia.org.txt
