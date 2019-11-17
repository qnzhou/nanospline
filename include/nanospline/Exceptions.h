#pragma once

#include <exception>
#include <string>

namespace nanospline {

class not_implemented_error : public std::exception {
    public:
        not_implemented_error() {}
        not_implemented_error(const std::string& reason) : m_reason(reason) {}

        const char* what() const noexcept override {
            if (m_reason.empty()) {
                return "This method is not implemented!";
            } else {
                return (std::string("This method is not implemented because... ")
                    + m_reason).c_str();
            }
        }

    private:
        std::string m_reason;
};

class invalid_setting_error : public std::exception {
    public:
        invalid_setting_error() {}
        invalid_setting_error(const std::string& reason) : m_reason(reason) {}

        const char* what() const noexcept override {
            if (m_reason.empty()) {
                return "Invalid setting!";
            } else {
                return (std::string("Invalid setting: ")
                    + m_reason).c_str();
            }
        }

    private:
        std::string m_reason;
};

}
